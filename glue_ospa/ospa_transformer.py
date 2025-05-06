# ospa_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from ospa_attention import OSPAMultiHeadAttention


class OSPATransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with OSPA attention mechanism."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", orth_mode="regularize", layer_norm_eps=1e-5):
        super(OSPATransformerEncoderLayer, self).__init__()
        
        # OSPA Multi-head attention
        self.self_attn = OSPAMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            orth_mode=orth_mode
        )
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = _get_activation_fn(activation)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: source sequence [seq_len, batch_size, embed_dim]
            src_mask: mask for src sequence [seq_len, seq_len]
            src_key_padding_mask: mask for src keys per batch [batch_size, seq_len]
        """
        # Multi-head attention block
        src2, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        
        # Add & Norm (first residual connection)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # Add & Norm (second residual connection)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def get_orthogonality_penalty(self):
        """Return the orthogonality penalty from the OSPA attention."""
        return self.self_attn.get_orthogonality_penalty()


class OSPATransformerEncoder(nn.Module):
    """TransformerEncoder with OSPA attention mechanism."""
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(OSPATransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output
    
    def get_orthogonality_penalty(self):
        """Calculate the total orthogonality penalty across all layers."""
        return sum(layer.get_orthogonality_penalty() for layer in self.layers)


class OSPATransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer with OSPA attention mechanism."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", orth_mode="regularize", layer_norm_eps=1e-5):
        super(OSPATransformerDecoderLayer, self).__init__()
        
        # OSPA Multi-head attentions
        self.self_attn = OSPAMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            orth_mode=orth_mode
        )
        self.multihead_attn = OSPAMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            orth_mode=orth_mode
        )
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        self.activation = _get_activation_fn(activation)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: target sequence [tgt_len, batch_size, embed_dim]
            memory: memory from encoder [src_len, batch_size, embed_dim]
            tgt_mask: mask for target sequence [tgt_len, tgt_len]
            memory_mask: mask for memory [tgt_len, src_len]
            tgt_key_padding_mask: mask for tgt keys per batch [batch_size, tgt_len]
            memory_key_padding_mask: mask for memory keys per batch [batch_size, src_len]
        """
        # Self attention block
        tgt2, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        
        # Add & Norm (first residual connection)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention block
        tgt2, _ = self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        
        # Add & Norm (second residual connection)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
        # Add & Norm (third residual connection)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
    
    def get_orthogonality_penalty(self):
        """Return the orthogonality penalty from both OSPA attentions."""
        return self.self_attn.get_orthogonality_penalty() + self.multihead_attn.get_orthogonality_penalty()


class OSPATransformerDecoder(nn.Module):
    """TransformerDecoder with OSPA attention mechanism."""
    
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(OSPATransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        
        for layer in self.layers:
            output = layer(
                output, 
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output
    
    def get_orthogonality_penalty(self):
        """Calculate the total orthogonality penalty across all layers."""
        return sum(layer.get_orthogonality_penalty() for layer in self.layers)


class OSPATransformer(nn.Module):
    """Transformer model with OSPA attention mechanism."""
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu", 
                 orth_mode="regularize", orth_penalty_weight=0.01):
        super(OSPATransformer, self).__init__()
        
        # Create encoder layer and stack
        encoder_layer = OSPATransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            orth_mode=orth_mode
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = OSPATransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # Create decoder layer and stack
        decoder_layer = OSPATransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            orth_mode=orth_mode
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = OSPATransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.d_model = d_model
        self.nhead = nhead
        self.orth_mode = orth_mode
        self.orth_penalty_weight = orth_penalty_weight
        
        # Initialize parameters
        self._reset_parameters()
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src: source sequence [src_len, batch_size, embed_dim]
            tgt: target sequence [tgt_len, batch_size, embed_dim]
            src_mask: mask for source [src_len, src_len]
            tgt_mask: mask for target [tgt_len, tgt_len]
            memory_mask: mask for encoder memory [tgt_len, src_len]
            src_key_padding_mask: key padding mask for source [batch_size, src_len]
            tgt_key_padding_mask: key padding mask for target [batch_size, tgt_len]
            memory_key_padding_mask: key padding mask for memory [batch_size, src_len]
        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        output = self.decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return output
    
    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """Encode the source sequence."""
        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Decode the target sequence given the memory from encoder."""
        return self.decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
    
    def get_orthogonality_penalty(self):
        """Calculate the total orthogonality penalty across both encoder and decoder."""
        return self.encoder.get_orthogonality_penalty() + self.decoder.get_orthogonality_penalty()
    
    def _reset_parameters(self):
        """Initialize parameters with Glorot uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# Helper functions
def _get_clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return activation function based on string name."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"Activation function {activation} not found.")