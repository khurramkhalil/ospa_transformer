#!/usr/bin/env python3
"""
Train a question answering model with OSPA attention.
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate

from src.models.transformer import OSPATransformer
from src.utils.logging import setup_logger

# QA model using OSPA
class QuestionAnsweringModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, enforce_mode='regularize',
                 ortho_penalty_weight=0.01, max_seq_length=512, pad_idx=0):
        super(QuestionAnsweringModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # OSPA Transformer (encoder-only)
        self.transformer = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enforce_mode=enforce_mode,
            ortho_penalty_weight=ortho_penalty_weight,
            max_seq_length=max_seq_length,
            use_decoder=False
        )
        
        # Output layers for start and end positions
        self.qa_outputs = nn.Linear(d_model, 2)
        
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
        import math
        src = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Pass through transformer
        encoded, attentions = self.transformer(
            src, src_key_padding_mask=src_key_padding_mask
        )
        
        # Get start/end logits
        logits = self.qa_outputs(encoded)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # Get orthogonality penalty
        ortho_penalty = self.transformer.get_orthogonality_penalty()
        
        return start_logits, end_logits, ortho_penalty

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create experiment directory
    experiment_dir = os.path.join(
        args.output_dir,
        f"squad_{args.enforce_mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger(experiment_dir)
    logger.info(f"Arguments: {args}")
    
    # Load SQuAD dataset
    logger.info("Loading SQuAD dataset...")
    squad_dataset = load_dataset("squad")
    
    # Implement preprocessing and training here
    logger.info("Dataset loaded successfully")
    
    # Create model
    logger.info("Creating OSPA model for question answering...")
    model = QuestionAnsweringModel(
        vocab_size=len(tokenizer),
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
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("SQUAD experiment setup complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OSPA for Question Answering")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store datasets")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Directory to save outputs")
    parser.add_argument("--max_seq_length", type=int, default=384,
                        help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model dimension")
    parser.add_argument("--nhead", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=3072,
                        help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--enforce_mode", type=str, default="regularize",
                        choices=["regularize", "strict", "init"],
                        help="How to enforce orthogonality")
    parser.add_argument("--ortho_penalty_weight", type=float, default=0.0005,
                        help="Weight for orthogonality penalty")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
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
