import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class VanillaMultiHeadAttention(nn.Module):
    """Standard multi-head attention as in the original Transformer paper."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, add_bias_kv=False, add_zero_attn=False):
        super(VanillaMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.add_zero_attn = add_zero_attn
        
        # Optional bias for key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)
        else:
            self.bias_k = self.bias_v = None
            
        self.dropout_p = dropout
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True):
        """
        Input shape: Time x Batch x Channel (T x B x C)
        or Batch x Time x Channel (B x T x C) if batch_first=True
        """
        # Default shapes are L x B x D (Length x Batch x Dimension)
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        # Apply linear projections to get queries, keys, and values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape q, k, v for multi-head attention
        # [L, B, D] -> [L, B, H, D/H] -> [B, H, L, D/H]
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        # If specified, add zero attention
        if self.add_zero_attn:
            zero_attn_shape = (bsz, self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=2)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=2)
            if attn_mask is not None:
                attn_mask = torch.cat([
                    attn_mask, 
                    torch.zeros(attn_mask.size(0), 1, dtype=attn_mask.dtype, device=attn_mask.device)
                ], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([
                    key_padding_mask, 
                    torch.zeros(key_padding_mask.size(0), 1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                ], dim=1)
                
        # Calculate attention scores
        # [B, H, L, D/H] x [B, H, D/H, S] = [B, H, L, S]
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [S, S] -> [1, 1, S, S]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, S, S] -> [B, 1, S, S]
            attn_weights = attn_weights + attn_mask
            
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        # Apply attention weights to values
        # [B, H, L, S] x [B, H, S, D/H] = [B, H, L, D/H]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output and apply output projection
        # [B, H, L, D/H] -> [L, B, D]
        attn_output = attn_output.transpose(1, 2).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            if average_attn_weights:
                # Average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


class LinformerSelfAttention(nn.Module):
    """
    Linformer attention with linear complexity in sequence length.
    Uses low-rank approximation of the attention matrix.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, k=256, bias=True):
        super(LinformerSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.k = k  # Low-rank dimension
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # E and F projection matrices for each head
        self.E = nn.Parameter(torch.Tensor(num_heads, k, self.head_dim))
        self.F = nn.Parameter(torch.Tensor(num_heads, k, self.head_dim))
        
        self.dropout_p = dropout
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, batch_idx_for_debug: int = -1):
        """
        Input shape: Time x Batch x Channel (T x B x C)
        """
        # Default shapes are L x B x D (Length x Batch x Dimension)
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        # Apply linear projections to get queries, keys, and values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape q, k, v for multi-head attention
        # [L, B, D] -> [L, B, H, D/H] -> [B, H, L, D/H]
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        
        # Linear attention approximation
        # Project keys and values to lower dimension using E and F
        # [B, H, S, D/H] -> [B, H, k]
        k_projected = torch.einsum('bhsd,hkd->bhsk', k, self.E)
        v_projected = torch.einsum('bhsd,hkd->bhsk', v, self.F)
        
        # Calculate attention scores using the projected keys
        # [B, H, L, D/H] x [B, H, D/H, k] = [B, H, L, k]
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling
        attn_weights = torch.matmul(q, k_projected)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        # Apply attention weights to projected values
        # [B, H, L, k] x [B, H, k, D/H] = [B, H, L, D/H]
        attn_output = torch.matmul(attn_weights, v_projected.transpose(-2, -1))
        
        # Reshape output and apply output projection
        # [B, H, L, D/H] -> [L, B, D]
        attn_output = attn_output.transpose(1, 2).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # Return attention output and weights
        if need_weights:
            # Note: Attention weights are approximated in Linformer
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


class VanillaTransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", layer_norm_eps=1e-5):
        super(VanillaTransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = VanillaMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
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


class LinformerTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Linformer attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", layer_norm_eps=1e-5, k=256):
        super(LinformerTransformerEncoderLayer, self).__init__()
        
        # Linformer multi-head attention
        self.self_attn = LinformerSelfAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            k=k
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


class VanillaTransformerEncoder(nn.Module):
    """Standard TransformerEncoder."""
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(VanillaTransformerEncoder, self).__init__()
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


class LinformerTransformerEncoder(nn.Module):
    """TransformerEncoder with Linformer attention."""
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(LinformerTransformerEncoder, self).__init__()
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


class VanillaTransformer(nn.Module):
    """Standard Transformer model."""
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(VanillaTransformer, self).__init__()
        
        # Create encoder layer and stack
        encoder_layer = VanillaTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = VanillaTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Initialize parameters
        self._reset_parameters()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: source sequence [src_len, batch_size, embed_dim]
            src_mask: mask for source [src_len, src_len]
            src_key_padding_mask: key padding mask for source [batch_size, src_len]
        """
        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    
    def _reset_parameters(self):
        """Initialize parameters with Glorot uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LinformerTransformer(nn.Module):
    """Transformer model with Linformer attention."""
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu", k=256):
        super(LinformerTransformer, self).__init__()
        
        # Create encoder layer and stack
        encoder_layer = LinformerTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            k=k
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = LinformerTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.d_model = d_model
        self.nhead = nhead
        self.k = k
        
        # Initialize parameters
        self._reset_parameters()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: source sequence [src_len, batch_size, embed_dim]
            src_mask: mask for source [src_len, src_len]
            src_key_padding_mask: key padding mask for source [batch_size, src_len]
        """
        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    
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