# ospa_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging # Add logging
from ospa_attention import OSPAMultiHeadAttention

logger = logging.getLogger(__name__) # Setup logger for this module

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
        
    # def forward(self, src, src_mask=None, src_key_padding_mask=None):
    #     """
    #     Args:
    #         src: source sequence [seq_len, batch_size, embed_dim]
    #         src_mask: mask for src sequence [seq_len, seq_len]
    #         src_key_padding_mask: mask for src keys per batch [batch_size, seq_len]
    #     """
    #     # Multi-head attention block
    #     src2, _ = self.self_attn(
    #         query=src,
    #         key=src,
    #         value=src,
    #         attn_mask=src_mask,
    #         key_padding_mask=src_key_padding_mask
    #     )
        
    #     # Add & Norm (first residual connection)
    #     src = src + self.dropout1(src2)
    #     src = self.norm1(src)
        
    #     # Feed forward block
    #     src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
    #     # Add & Norm (second residual connection)
    #     src = src + self.dropout2(src2)
    #     src = self.norm2(src)
        
    #     return src

# In ospa_transformer.py (if OSPA is being tested)
# OR if you have a custom VanillaTransformerEncoderLayer

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # --- DEBUG: Input to layer ---
        if torch.isnan(src).any():
            logger.error(f"!!! LAYER INPUT IS NAN !!! shape: {src.shape}")
            # This shouldn't happen if prev layer output was okay, but good check
        # logger.info(f"Layer Input - min={src.min():.2e}, max={src.max():.2e}, mean={src.mean():.2e}, has_nan={torch.isnan(src).any()}")

        # --- Multi-head attention block ---
        # Further debug inside self_attn if needed
        # logger.info(f"Before Self-Attn Q - min={src.min():.2e}, max={src.max():.2e}") # Same as src
        src2_attn_output, attn_weights = self.self_attn( # Assuming self_attn returns weights for debug
            query=src, key=src, value=src,
            attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
            need_weights=True # Ensure your MHA can return weights
        )
        if torch.isnan(src2_attn_output).any():
            logger.error(f"!!! NAN after self_attn block !!!")
            logger.error(f"  Attn Input (src) - min={src.min():.2e}, max={src.max():.2e}, mean={src.mean():.2e}, has_nan={torch.isnan(src).any()}")
            if attn_weights is not None: # Check attention weights themselves
                logger.error(f"  Attn Weights - min={attn_weights.min():.2e}, max={attn_weights.max():.2e}, mean={attn_weights.mean():.2e}, has_nan={torch.isnan(attn_weights).any()}")
            # You might want to inspect Q, K, V from within self_attn if this is where NaNs appear
            raise ValueError("NaN from self_attn_output") # Stop execution here to inspect
        # logger.info(f"After Self-Attn - min={src2_attn_output.min():.2e}, max={src2_attn_output.max():.2e}, mean={src2_attn_output.mean():.2e}, has_nan={torch.isnan(src2_attn_output).any()}")


        # --- Add & Norm (first residual connection) ---
        src_after_dropout1 = self.dropout1(src2_attn_output)
        if torch.isnan(src_after_dropout1).any():
            logger.error(f"!!! NAN after self_attn_dropout !!!")
            raise ValueError("NaN from self_attn_dropout")
        
        src_res1 = src + src_after_dropout1
        if torch.isnan(src_res1).any():
            logger.error(f"!!! NAN after first residual add !!!")
            logger.error(f"  src             - min={src.min():.2e}, max={src.max():.2e}, has_nan={torch.isnan(src).any()}")
            logger.error(f"  src_after_drop1 - min={src_after_dropout1.min():.2e}, max={src_after_dropout1.max():.2e}, has_nan={torch.isnan(src_after_dropout1).any()}")
            raise ValueError("NaN from first residual add")

        src_norm1 = self.norm1(src_res1)
        if torch.isnan(src_norm1).any():
            logger.error(f"!!! NAN after first LayerNorm (norm1) !!!")
            logger.error(f"  Input to norm1 (src_res1) - min={src_res1.min():.2e}, max={src_res1.max():.2e}, mean={src_res1.mean():.2e}, has_nan={torch.isnan(src_res1).any()}")
            # Check LayerNorm weights/bias
            logger.error(f"  norm1.weight - min={self.norm1.weight.min():.2e}, max={self.norm1.weight.max():.2e}")
            logger.error(f"  norm1.bias   - min={self.norm1.bias.min():.2e}, max={self.norm1.bias.max():.2e}")
            raise ValueError("NaN from first LayerNorm")
        # logger.info(f"After Norm1 - min={src_norm1.min():.2e}, max={src_norm1.max():.2e}, mean={src_norm1.mean():.2e}, has_nan={torch.isnan(src_norm1).any()}")


        # --- Feed forward block ---
        ff_hidden = self.activation(self.linear1(src_norm1))
        if torch.isnan(ff_hidden).any():
            logger.error(f"!!! NAN after FFN activation(linear1) !!!")
            logger.error(f"  Input to FFN (src_norm1) - min={src_norm1.min():.2e}, max={src_norm1.max():.2e}, mean={src_norm1.mean():.2e}, has_nan={torch.isnan(src_norm1).any()}")
            logger.error(f"  linear1.weight norm: {self.linear1.weight.norm().item():.2e}")
            raise ValueError("NaN from FFN linear1/activation")
        
        src2_ff_output = self.linear2(self.dropout(ff_hidden))
        if torch.isnan(src2_ff_output).any():
            logger.error(f"!!! NAN after FFN linear2(dropout(activation(linear1))) !!!")
            raise ValueError("NaN from FFN linear2/dropout")
        # logger.info(f"After FFN - min={src2_ff_output.min():.2e}, max={src2_ff_output.max():.2e}, mean={src2_ff_output.mean():.2e}, has_nan={torch.isnan(src2_ff_output).any()}")


        # --- Add & Norm (second residual connection) ---
        src_after_dropout2 = self.dropout2(src2_ff_output)
        if torch.isnan(src_after_dropout2).any():
            logger.error(f"!!! NAN after FFN_dropout !!!")
            raise ValueError("NaN from FFN_dropout")

        src_res2 = src_norm1 + src_after_dropout2 # Residual from AFTER first norm
        if torch.isnan(src_res2).any():
            logger.error(f"!!! NAN after second residual add !!!")
            raise ValueError("NaN from second residual add")

        src_norm2 = self.norm2(src_res2)
        if torch.isnan(src_norm2).any():
            logger.error(f"!!! NAN after second LayerNorm (norm2) !!!")
            logger.error(f"  Input to norm2 (src_res2) - min={src_res2.min():.2e}, max={src_res2.max():.2e}, mean={src_res2.mean():.2e}, has_nan={torch.isnan(src_res2).any()}")
            raise ValueError("NaN from second LayerNorm")
        # logger.info(f"Layer Output - min={src_norm2.min():.2e}, max={src_norm2.max():.2e}, mean={src_norm2.mean():.2e}, has_nan={torch.isnan(src_norm2).any()}")

        return src_norm2

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