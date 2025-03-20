"""
Transformer model using Orthogonal Subspace Projection Attention (OSPA).

This module implements a complete transformer architecture using OSPA attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, List, Tuple, Union

from .ospa import OSPAEncoderLayer, OSPADecoderLayer


class OSPATransformerEncoder(nn.Module):
    """Transformer encoder with OSPA attention."""
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        Initialize OSPA transformer encoder.
        
        Args:
            encoder_layer: An instance of OSPAEncoderLayer
            num_layers: Number of encoder layers
            norm: Optional normalization layer
        """
        super(OSPATransformerEncoder, self).__init__()
        
        # Create multiple identical encoder layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass for OSPA transformer encoder.
        
        Args:
            src: Input sequence of shape (batch_size, seq_len, d_model)
            mask: Attention mask
            src_key_padding_mask: Key padding mask
            
        Returns:
            Output sequence and attention weights from each layer
        """
        output = src
        all_attentions = []
        
        for layer in self.layers:
            output, attn_weights = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            all_attentions.append(attn_weights)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output, all_attentions
    
    def get_orthogonality_penalty(self):
        """Get total orthogonality penalty across all encoder layers."""
        return sum(layer.get_orthogonality_penalty() for layer in self.layers)


class OSPATransformerDecoder(nn.Module):
    """Transformer decoder with OSPA attention."""
    
    def __init__(self, decoder_layer, num_layers, norm=None):
        """
        Initialize OSPA transformer decoder.
        
        Args:
            decoder_layer: An instance of OSPADecoderLayer
            num_layers: Number of decoder layers
            norm: Optional normalization layer
        """
        super(OSPATransformerDecoder, self).__init__()
        
        # Create multiple identical decoder layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for OSPA transformer decoder.
        
        Args:
            tgt: Target sequence of shape (batch_size, tgt_len, d_model)
            memory: Memory from encoder of shape (batch_size, src_len, d_model)
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Output sequence and attention weights from each layer
        """
        output = tgt
        all_self_attentions = []
        all_cross_attentions = []
        
        for layer in self.layers:
            output, (self_attn, cross_attn) = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            all_self_attentions.append(self_attn)
            all_cross_attentions.append(cross_attn)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output, (all_self_attentions, all_cross_attentions)
    
    def get_orthogonality_penalty(self):
        """Get total orthogonality penalty across all decoder layers."""
        return sum(layer.get_orthogonality_penalty() for layer in self.layers)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class OSPATransformer(nn.Module):
    """
    Full transformer model with OSPA attention.
    
    This can be used for sequence-to-sequence tasks or encoder-only tasks.
    """
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu", enforce_mode='regularize',
                 ortho_penalty_weight=0.01, norm_first=False, use_decoder=True,
                 max_seq_length=5000, device=None, dtype=None):
        """
        Initialize OSPA transformer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
            enforce_mode: How to enforce orthogonality
            ortho_penalty_weight: Weight for orthogonality penalty
            norm_first: If True, use pre-norm architecture
            use_decoder: Whether to include a decoder (for seq2seq tasks)
            max_seq_length: Maximum sequence length for positional encoding
            device: Device for computation
            dtype: Data type for computation
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OSPATransformer, self).__init__()
        
        # Save configuration
        self.d_model = d_model
        self.nhead = nhead
        self.use_decoder = use_decoder
        self.ortho_penalty_weight = ortho_penalty_weight
        self.enforce_mode = enforce_mode
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Create encoder
        encoder_layer = OSPAEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, enforce_mode, norm_first, device=device, dtype=dtype
        )
        encoder_norm = nn.LayerNorm(d_model, **factory_kwargs)
        self.encoder = OSPATransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # Create decoder if needed
        if use_decoder:
            decoder_layer = OSPADecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation, enforce_mode, norm_first, device=device, dtype=dtype
            )
            decoder_norm = nn.LayerNorm(d_model, **factory_kwargs)
            self.decoder = OSPATransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        else:
            self.decoder = None
    
    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Encode input sequence.
        
        Args:
            src: Input sequence of shape (batch_size, seq_len, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Key padding mask
            
        Returns:
            Encoded sequence and attention weights
        """
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through encoder
        memory, encoder_attentions = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        
        return memory, encoder_attentions
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
              tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Decode encoded sequence.
        
        Args:
            tgt: Target sequence of shape (batch_size, tgt_len, d_model)
            memory: Memory from encoder
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Decoded sequence and attention weights
        """
        if not self.use_decoder:
            raise ValueError("Model was configured without a decoder.")
        
        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        
        # Pass through decoder
        output, (self_attentions, cross_attentions) = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return output, (self_attentions, cross_attentions)
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None,
               memory_mask=None, src_key_padding_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for OSPA transformer.
        
        For encoder-only tasks, provide only src.
        For seq2seq tasks, provide both src and tgt.
        
        Args:
            src: Source sequence
            tgt: Optional target sequence (for seq2seq tasks)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            src_key_padding_mask: Source key padding mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Output sequence and attention weights
        """
        # Encode source sequence
        memory, encoder_attentions = self.encode(
            src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        
        # If decoder is used and target is provided, decode
        if self.use_decoder and tgt is not None:
            output, (decoder_self_attentions, decoder_cross_attentions) = self.decode(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            return output, {
                'encoder_attentions': encoder_attentions,
                'decoder_self_attentions': decoder_self_attentions,
                'decoder_cross_attentions': decoder_cross_attentions
            }
        else:
            # Encoder-only model
            return memory, {'encoder_attentions': encoder_attentions}
    
    def get_orthogonality_penalty(self):
        """
        Calculate total orthogonality penalty for the model.
        
        This is used for regularization during training.
        """
        penalty = self.encoder.get_orthogonality_penalty()
        
        if self.use_decoder:
            penalty += self.decoder.get_orthogonality_penalty()
        
        return self.ortho_penalty_weight * penalty