#!/usr/bin/env python3
"""
Train a translation model with OSPA attention.

This script trains a machine translation model using OSPA transformer
on the IWSLT dataset.
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import MarianTokenizer
from datasets import load_dataset, load_from_disk
import evaluate

from src.models.transformer import OSPATransformer
from src.utils.logging import setup_logger


class TranslationModel(nn.Module):
    """Translation model using OSPA Transformer."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, enforce_mode='regularize',
                 ortho_penalty_weight=0.01, max_seq_length=128, pad_idx=1):
        super(TranslationModel, self).__init__()
        
        # Source and target embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # OSPA Transformer (encoder-decoder)
        self.transformer = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enforce_mode=enforce_mode,
            ortho_penalty_weight=ortho_penalty_weight,
            max_seq_length=max_seq_length,
            use_decoder=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Save configuration
        self.d_model = d_model
        self.pad_idx = pad_idx
    
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Create masks
        if src_mask is None:
            src_key_padding_mask = (src_ids == self.pad_idx)
        else:
            src_key_padding_mask = src_mask
        
        if tgt_mask is None:
            tgt_key_padding_mask = (tgt_ids[:, :-1] == self.pad_idx)
        else:
            tgt_key_padding_mask = tgt_mask[:, :-1]
        
        # Create causal mask for decoder
        tgt_len = tgt_ids.size(1) - 1
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt_ids.device) == 1
        ).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(
            causal_mask == 0, float('-inf')).masked_fill(
            causal_mask == 1, float(0.0)
        )
        
        # Convert to embeddings
        src_emb = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_ids[:, :-1]) * math.sqrt(self.d_model)
        
        # Pass through transformer
        output, attns = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=causal_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        # Get orthogonality penalty
        ortho_penalty = self.transformer.get_orthogonality_penalty()
        
        return logits, ortho_penalty


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        src_ids = batch['source_ids'].to(device)
        tgt_ids = batch['target_ids'].to(device)
        src_mask = batch['source_mask'].to(device)
        tgt_mask = batch['target_mask'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, ortho_penalty = model(src_ids, tgt_ids, src_mask, tgt_mask)
        
        # Calculate loss
        # Reshape logits and targets for loss calculation
        # logits: [batch_size, tgt_len-1, vocab_size]
        # targets: [batch_size, tgt_len-1]
        logits = logits.contiguous().view(-1, logits.size(-1))
        targets = tgt_ids[:, 1:].contiguous().view(-1)
        
        # Compute loss only on non-padded tokens
        loss_mask = targets != model.pad_idx
        targets = targets[loss_mask]
        logits = logits[loss_mask]
        
        if targets.size(0) > 0:  # Skip if all targets are padding
            translation_loss = criterion(logits, targets)
            loss = translation_loss + ortho_penalty
            
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
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item() if targets.size(0) > 0 else 0})
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            src_ids = batch['source_ids'].to(device)
            tgt_ids = batch['target_ids'].to(device)
            src_mask = batch['source_mask'].to(device)
            tgt_mask = batch['target_mask'].to(device)
            
            # Forward pass
            logits, ortho_penalty = model(src_ids, tgt_ids, src_mask, tgt_mask)
            
            # Calculate loss
            logits = logits.contiguous().view(-1, logits.size(-1))
            targets = tgt_ids[:, 1:].contiguous().view(-1)
            
            # Compute loss only on non-padded tokens
            loss_mask = targets != model.pad_idx
            targets = targets[loss_mask]
            logits = logits[loss_mask]
            
            if targets.size(0) > 0:  # Skip if all targets are padding
                translation_loss = criterion(logits, targets)
                loss = translation_loss + ortho_penalty
                
                # Update metrics
                total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


class TranslationDataset(torch.utils.data.Dataset):
    """Dataset for machine translation."""
    
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, max_length=128):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get source and target text
        source_text = item['translation']['de']
        target_text = item['translation']['en']
        
        # Tokenize
        source_tokens = self.src_tokenizer.encode(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).squeeze()
        
        target_tokens = self.tgt_tokenizer.encode(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).squeeze()
        
        # Create attention masks
        source_mask = source_tokens == self.src_tokenizer.pad_token_id
        target_mask = target_tokens == self.tgt_tokenizer.pad_token_id
        
        return {
            'source_ids': source_tokens,
            'target_ids': target_tokens,
            'source_mask': source_mask,
            'target_mask': target_mask,
            'source_text': source_text,
            'target_text': target_text
        }


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load tokenizers
    print("Loading tokenizers...")
    if args.model_name:
        src_tokenizer = MarianTokenizer.from_pretrained(args.model_name)
        tgt_tokenizer = src_tokenizer
    else:
        # Simple fallback if custom tokenizers are needed
        from transformers import AutoTokenizer
        src_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        tgt_tokenizer = src_tokenizer
    
    # Create experiment directory
    experiment_dir = os.path.join(
        args.output_dir,
        f"iwslt_{args.enforce_mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger(experiment_dir)
    logger.info(f"Arguments: {args}")
    
    # Load dataset
    print("Loading dataset...")
    try:
        iwslt_dataset = load_from_disk(os.path.join(args.data_dir, "de-en"))
    except:
        logger.info("Dataset not found locally, loading from HuggingFace")
        iwslt_dataset = load_dataset("iwslt2017", "iwslt2017-de-en")
    
    # Create train and validation datasets
    train_dataset = TranslationDataset(
        iwslt_dataset['train'],
        src_tokenizer,
        tgt_tokenizer,
        max_length=args.max_seq_length
    )
    
    val_dataset = TranslationDataset(
        iwslt_dataset['validation'],
        src_tokenizer,
        tgt_tokenizer,
        max_length=args.max_seq_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Creating OSPA Transformer model...")
    model = TranslationModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        enforce_mode=args.enforce_mode,
        ortho_penalty_weight=args.ortho_penalty_weight,
        max_seq_length=args.max_seq_length,
        pad_idx=src_tokenizer.pad_token_id
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
    best_val_loss = float('inf')
    all_metrics = []
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(
            model, val_loader, criterion, device
        )
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # Save metrics for this epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        all_metrics.append(epoch_metrics)
        
        # Save all metrics to a file
        with open(os.path.join(experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model
            model_path = os.path.join(experiment_dir, "best_model.pt")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with validation loss = {val_loss:.4f}")
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OSPA Transformer for Translation")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data/iwslt",
                        help="Directory containing the datasets")
    parser.add_argument("--output_dir", type=str, default="./experiments/iwslt",
                        help="Directory to save the outputs")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-de-en",
                        help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6,
                        help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6,
                        help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048,
                        help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--enforce_mode", type=str, default="regularize",
                        choices=["regularize", "init"],
                        help="How to enforce orthogonality")
    parser.add_argument("--ortho_penalty_weight", type=float, default=0.01,
                        help="Weight for orthogonality penalty")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    main(args)
