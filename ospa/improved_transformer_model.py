# improved_transformer_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming ospa_transformer.py exists and defines OSPATransformer
from ospa_transformer import OSPATransformer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embed_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container for a Transformer model with token embeddings and task-specific heads."""

    def __init__(self, transformer_type, vocab_size, d_model, nhead, nlayers,
                 dropout=0.1, dim_feedforward=2048, orth_mode="regularize",
                 orth_penalty_weight=0.01, task="lm"):
        super(TransformerModel, self).__init__()
        self.transformer_type = transformer_type
        self.d_model = d_model
        self.orth_mode = orth_mode
        self.orth_penalty_weight = orth_penalty_weight # Stored but primarily used by OSPATransformer internally
        self.task = task
        self.vocab_size = vocab_size

        # Token embedding and positional encoding
        self.encoder = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Initialize the appropriate transformer architecture
        if transformer_type == "ospa":
            self.transformer = OSPATransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,  # Use only encoder for these tasks
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                orth_mode=orth_mode,
                # orth_penalty_weight passed but handled internally via get_orthogonality_penalty
            )
        elif transformer_type == "vanilla":
            # Ensure VanillaTransformer is correctly imported or defined
            from baseline_models import VanillaTransformer # Example import
            self.transformer = VanillaTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        elif transformer_type == "linformer":
             # Ensure LinformerTransformer is correctly imported or defined
            from baseline_models import LinformerTransformer # Example import
            self.transformer = LinformerTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                k=256  # Example projection dimension for Linformer
            )
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")

        # Task-specific output heads
        if task == "lm":  # Language modeling
            self.output_scaling = 0.1  # Optional scaling factor
            self.output_layer = nn.Linear(d_model, self.vocab_size)
        elif task == "classification":  # Text classification
            self.output_layer = nn.Linear(d_model, 2)  # Assuming binary classification
        else:
            raise ValueError(f"Unknown task: {task}")

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        initrange = 0.02 # Slightly larger range sometimes helps
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

        if hasattr(self, 'output_layer'):
            # Initialize output projection
            if self.task == "lm":
                # Use Xavier initialization, potentially with small gain
                nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1) # Adjusted gain slightly
                if self.output_layer.bias is not None:
                    nn.init.zeros_(self.output_layer.bias)
            else: # Classification
                nn.init.uniform_(self.output_layer.weight, -initrange, initrange)
                if self.output_layer.bias is not None:
                     nn.init.zeros_(self.output_layer.bias)


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Token indices [seq_len, batch_size]
            src_mask: Mask for self-attention [seq_len, seq_len] (e.g., causal mask)
            src_key_padding_mask: Mask for padding tokens [batch_size, seq_len]
        """
        # 1. Embedding and Positional Encoding
        # Apply sqrt(d_model) scaling *after* embedding
        src_embed = self.encoder(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embed)

        if torch.isnan(src_pos).any():
            print("WARNING: NaN detected after embedding/pos encoding.")
            # Handle appropriately, e.g., return zero output or raise error
            # return torch.zeros_like(src_pos) # Example handling
            raise ValueError("NaN values detected in embeddings/positional encoding")

        # 2. Pass through Transformer Encoder
        # Check if the underlying transformer has an 'encoder' attribute (like OSPATransformer)
        # or if it's the encoder itself (like potentially VanillaTransformer)
        if hasattr(self.transformer, 'encoder') and isinstance(self.transformer.encoder, nn.Module):
            transformer_output = self.transformer.encoder(
                src_pos,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        # If the transformer object itself is the encoder stack (common pattern)
        elif isinstance(self.transformer, nn.TransformerEncoder) or \
             (hasattr(self.transformer, 'layers') and isinstance(self.transformer.layers, nn.ModuleList)): # Heuristic check
             transformer_output = self.transformer(
                 src_pos,
                 mask=src_mask,
                 src_key_padding_mask=src_key_padding_mask
             )
        else:
             # Fallback or error if structure is unknown
             # transformer_output = self.transformer(src_pos) # Simplest assumption
             raise TypeError(f"Unsure how to call the underlying transformer of type {type(self.transformer)}. Implement specific handling.")


        if torch.isnan(transformer_output).any():
            print("WARNING: NaN detected after transformer layers.")
            raise ValueError("NaN values detected in transformer output")


        # 3. Task-specific Output Layer
        if self.task == "lm":
            # Apply optional scaling before final projection
            scaled_output = transformer_output * getattr(self, 'output_scaling', 1.0) # Use getattr for safety
            output = self.output_layer(scaled_output)
        elif self.task == "classification":
            # Use representation of the first token (e.g., [CLS])
            # Input to output_layer should be [batch_size, d_model]
            first_token_output = transformer_output[0, :, :] # Shape [batch_size, d_model]
            output = self.output_layer(first_token_output)
        else:
             # Should have been caught in init, but defensive check
             raise ValueError(f"Unknown task in forward pass: {self.task}")


        return output

    def get_orthogonality_penalty(self):
        """Delegates calculation to the underlying OSPA transformer if applicable."""
        if self.transformer_type == "ospa" and hasattr(self.transformer, 'get_orthogonality_penalty'):
            return self.transformer.get_orthogonality_penalty()
        return 0.0