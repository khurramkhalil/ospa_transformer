import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from orthogonal_linear import OrthogonalLinear
from ospa_attention import OSPAMultiHeadAttention
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
        self.orth_penalty_weight = orth_penalty_weight
        self.task = task
        
        # Token embedding and positional encoding
        self.encoder = nn.Embedding(vocab_size, d_model)
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
                orth_penalty_weight=orth_penalty_weight
            )
        elif transformer_type == "vanilla":
            from baseline_models import VanillaTransformer
            self.transformer = VanillaTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        elif transformer_type == "linformer":
            from baseline_models import LinformerTransformer
            self.transformer = LinformerTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                k=256  # Projection dimension for Linformer
            )
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
        
        # Task-specific output heads
        if task == "lm":  # Language modeling
            # Use adaptive softmax for more stable vocabulary projection
            self.output_scaling = 0.1  # Scaling factor to reduce logit magnitudes
            # Use a more carefully initialized output layer
            self.output_layer = nn.Linear(d_model, vocab_size)
        elif task == "classification":  # Text classification
            self.output_layer = nn.Linear(d_model, 2)  # Binary classification (e.g., IMDB)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights with appropriate scaling factors."""
        # Initialize embeddings with small values
        initrange = 0.01  # Reduced from typical 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        
        # Initialize output projection very carefully for stability
        if self.task == "lm":
            # Use Xavier initialization with small gain for large vocabulary
            nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01)
            nn.init.zeros_(self.output_layer.bias)
        else:
            # For classification, standard initialization is fine
            nn.init.uniform_(self.output_layer.weight, -initrange, initrange)
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Token indices [seq_len, batch_size]
            src_mask: Mask for self-attention [seq_len, seq_len]
            src_key_padding_mask: Mask for padding tokens [batch_size, seq_len]
        """
        # Token embedding and positional encoding with proper scaling
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Check for NaN values early
        if torch.isnan(src).any():
            raise ValueError("NaN values detected in embeddings")
            
        # Pass through transformer (only using encoder part)
        if hasattr(self.transformer, 'encoder'):
            output = self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Apply output layer based on task with scaling for stability
        if self.task == "lm":
            # Apply scaling before final projection to reduce logit magnitude
            scaled_output = output * self.output_scaling
            output = self.output_layer(scaled_output)
        elif self.task == "classification":
            # For classification, use the representation of the first token
            output = self.output_layer(output[0])
        
        return output
    
    def get_orthogonality_penalty(self):
        """Get the orthogonality penalty if using OSPA."""
        if self.transformer_type == "ospa":
            return self.transformer.get_orthogonality_penalty()
        return 0.0