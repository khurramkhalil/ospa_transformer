#!/usr/bin/env python3
"""
Training script for OSPA Transformer on GLUE tasks using the datasets library.

This script provides a complete training pipeline for OSPA on text classification.
"""

import os
import argparse
import time
import math
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import evaluate

# Import project modules
from src.models.transformer import OSPATransformer
from src.utils.logging import setup_logger


class SequenceClassifier(nn.Module):
    """Sequence classifier using OSPA Transformer."""
    
    def __init__(self, vocab_size, num_classes, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, enforce_mode='regularize',
                 ortho_penalty_weight=0.01, max_seq_length=512, pad_idx=0):
        super(SequenceClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # OSPA Transformer (encoder-only)
        self.transformer = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # No decoder for classification
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enforce_mode=enforce_mode,
            ortho_penalty_weight=ortho_penalty_weight,
            max_seq_length=max_seq_length,
            use_decoder=False
        )
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Save configuration
        self.d_model = d_model
        self.pad_idx = pad_idx
    
    def forward(self, input_ids, attention_mask=None):
        # Create padding mask for transformer
        if attention_mask is None:
            src_key_padding_mask = (input_ids == self.pad_idx)
        else:
            src_key_padding_mask = attention_mask.eq(0)
        
        # Convert input to embeddings
        src = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Pass through transformer
        encoded, attentions = self.transformer(
            src, src_key_padding_mask=src_key_padding_mask
        )
        
        # Use [CLS] token (first token) for classification
        cls_representation = encoded[:, 0]
        
        # Classify
        logits = self.classifier(cls_representation)
        
        # Get orthogonality penalty
        ortho_penalty = self.transformer.get_orthogonality_penalty()
        
        return logits, ortho_penalty


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, ortho_penalty = model(input_ids, attention_mask)
        
        # Calculate loss
        classification_loss = criterion(logits, labels)
        loss = classification_loss + ortho_penalty
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        batch_correct = (predicted == labels).sum().item()
        batch_total = labels.size(0)
        
        correct += batch_correct
        total += batch_total
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': batch_correct / batch_total
        })
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device, metric_name=None):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # Forward pass
            logits, ortho_penalty = model(input_ids, attention_mask)
            
            # Calculate loss
            classification_loss = criterion(logits, labels)
            loss = classification_loss + ortho_penalty
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            
            correct += batch_correct
            total += batch_total
            
            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    metrics = {
        'accuracy': correct / total
    }
    
    # Calculate additional metrics if provided
    if metric_name and metric_name.lower() != 'accuracy':
        try:
            metric = evaluate.load('glue', metric_name.lower())
            eval_result = metric.compute(predictions=np.array(all_preds), 
                                        references=np.array(all_labels))
            metrics.update(eval_result)
        except Exception as e:
            print(f"Error loading metric {metric_name}: {e}")
    
    return avg_loss, metrics


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get task info
    task_to_keys = {
        'sst2': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'qqp': ('question1', 'question2'),
    }
    
    if args.task not in task_to_keys:
        raise ValueError(f"Task {args.task} not supported")
    
    sentence1_key, sentence2_key = task_to_keys[args.task]
    
    # Determine number of labels
    task_to_labels = {
        'sst2': 2,
        'mnli': 3,
        'qqp': 2
    }
    num_labels = task_to_labels[args.task]
    
    # Create experiment directory
    experiment_dir = os.path.join(
        args.output_dir,
        f"{args.task}_{args.enforce_mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger(experiment_dir)
    logger.info(f"Arguments: {args}")
    
    # Load dataset
    logger.info(f"Loading {args.task} dataset...")
    raw_datasets = load_dataset("glue", args.task)
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Extract inputs
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else
            (examples[sentence1_key], examples[sentence2_key])
        )
        
        # Tokenize
        result = tokenizer(*texts, padding="max_length", max_length=args.max_seq_length, truncation=True)
        
        # Add labels
        if "label" in examples:
            result["labels"] = examples["label"]
        
        return result
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Preprocessing datasets",
    )
    
    # Get train/validation splits
    train_dataset = processed_datasets["train"]
    if args.task == "mnli":
        validation_dataset = processed_datasets["validation_matched"]
    else:
        validation_dataset = processed_datasets["validation"]
    
    # Convert to PyTorch tensors
    logger.info("Preparing data loaders...")
    
    # Training data
    train_input_ids = torch.tensor(train_dataset["input_ids"], dtype=torch.long)
    train_attention_mask = torch.tensor(train_dataset["attention_mask"], dtype=torch.long)
    train_labels = torch.tensor(train_dataset["labels"], dtype=torch.long)
    train_tensor_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    
    # Validation data
    val_input_ids = torch.tensor(validation_dataset["input_ids"], dtype=torch.long)
    val_attention_mask = torch.tensor(validation_dataset["attention_mask"], dtype=torch.long)
    val_labels = torch.tensor(validation_dataset["labels"], dtype=torch.long)
    val_tensor_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        val_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating OSPA Transformer model...")
    model = SequenceClassifier(
        vocab_size=len(tokenizer),
        num_classes=num_labels,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        enforce_mode=args.enforce_mode,
        ortho_penalty_weight=args.ortho_penalty_weight,
        max_seq_length=args.max_seq_length,
        pad_idx=tokenizer.pad_token_id
    )
    model.to(device)
    
    # Log model size
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Define scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate / 100
    )
    
    # Save configuration for reproducibility
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train and evaluate
    logger.info("Starting training...")
    best_eval_metric = 0.0
    all_metrics = []
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate
        eval_loss, eval_metrics = evaluate(
            model, eval_loader, criterion, device, args.task
        )
        
        # Log metrics
        log_str = f"Eval Loss: {eval_loss:.4f}"
        for metric_name, metric_value in eval_metrics.items():
            log_str += f", {metric_name.capitalize()}: {metric_value:.4f}"
        logger.info(log_str)
        
        # Save metrics for this epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'eval_loss': eval_loss,
            **eval_metrics
        }
        all_metrics.append(epoch_metrics)
        
        # Save all metrics to a file
        with open(os.path.join(experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save best model
        current_metric = eval_metrics.get('accuracy', 0.0)
        if current_metric > best_eval_metric:
            best_eval_metric = current_metric
            # Save model
            model_path = os.path.join(experiment_dir, "best_model.pt")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with accuracy = {current_metric:.4f}")
    
    logger.info(f"Training completed. Best accuracy: {best_eval_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OSPA Transformer on GLUE")
    
    # Data arguments
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mnli", "qqp"],
                        help="GLUE task to train on")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store the datasets")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Directory to save the outputs")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="Tokenizer to use")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048,
                        help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--enforce_mode", type=str, default="regularize",
                        choices=["regularize", "strict", "init"],
                        help="How to enforce orthogonality")
    parser.add_argument("--ortho_penalty_weight", type=float, default=0.01,
                        help="Weight for orthogonality penalty")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    main(args)