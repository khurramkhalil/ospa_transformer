"""
Training script for OSPA Transformer on language tasks.

This script demonstrates how to train and evaluate OSPA Transformer
on a sequence classification task from the GLUE benchmark.
"""

import os
import argparse
import time
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer, GlueDataset, GlueDataTrainingArguments
from datasets import load_dataset, load_metric

from src.models.transformer import OSPATransformer
import src.utils.logging as logging


class SequenceClassifier(nn.Module):
    """
    Sequence classifier using OSPA Transformer.
    
    This model uses an encoder-only OSPA Transformer for sequence classification.
    """
    
    def __init__(self, vocab_size, num_classes, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, enforce_mode='regularize',
                 ortho_penalty_weight=0.01, max_seq_length=512, pad_idx=0):
        """
        Initialize sequence classifier.
        
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            enforce_mode: How to enforce orthogonality
            ortho_penalty_weight: Weight for orthogonality penalty
            max_seq_length: Maximum sequence length
            pad_idx: Padding token index
        """
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
        """
        Forward pass for sequence classification.
        
        Args:
            input_ids: Input token indices
            attention_mask: Attention mask (1 for tokens, 0 for padding)
            
        Returns:
            logits: Classification logits
            ortho_penalty: Orthogonality penalty
        """
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
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device for computation
        scheduler: Learning rate scheduler
        
    Returns:
        avg_loss: Average loss for the epoch
        avg_acc: Average accuracy for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
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
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device for computation
        metric_name: Name of the metric to compute
        
    Returns:
        avg_loss: Average loss
        metrics: Dictionary of metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
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
            metric = load_metric('glue', metric_name.lower())
            eval_result = metric.compute(predictions=np.array(all_preds), 
                                         references=np.array(all_labels))
            metrics.update(eval_result)
        except:
            print(f"Error loading metric {metric_name}")
    
    return avg_loss, metrics


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Load dataset
    data_args = GlueDataTrainingArguments(
        task_name=args.task,
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length
    )
    
    # Get number of labels for the task
    if args.task == 'sst2':
        num_labels = 2
    elif args.task == 'mnli':
        num_labels = 3
    elif args.task == 'qqp':
        num_labels = 2
    else:
        raise ValueError(f"Task {args.task} not supported")
    
    # Create datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
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
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=args.learning_rate / 100
    )
    
    # Create experiment directory
    experiment_dir = os.path.join(
        args.output_dir,
        f"{args.task}_{args.enforce_mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize logger
    logger = logging.setup_logger(experiment_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Model: {model}")
    
    # Train and evaluate
    best_eval_metric = 0.0
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
        
        # Save best model
        # Choose appropriate main metric based on the task
        if args.task == 'sst2':
            main_metric_name = 'accuracy'
        elif args.task == 'mnli':
            main_metric_name = 'accuracy'
        else:
            main_metric_name = 'accuracy'  # Default
        
        current_metric = eval_metrics.get(main_metric_name, 0.0)
        if current_metric > best_eval_metric:
            best_eval_metric = current_metric
            # Save model
            model_path = os.path.join(experiment_dir, "best_model.pt")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with {main_metric_name} = {current_metric:.4f}")
    
    logger.info(f"Training completed. Best {main_metric_name}: {best_eval_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OSPA Transformer")
    
    # Data arguments
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mnli", "qqp"],
                        help="GLUE task to train on")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the datasets")
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
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    main(args)