"""
Orthogonal Subspace Projection Attention (OSPA) implementation.

This module implements the OSPA mechanism for transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def orthogonalize(weight, eps=1e-8):
    """
    Apply Gram-Schmidt process to orthogonalize weight matrix.
    
    Args:
        weight: Tensor of shape (out_features, in_features)
        eps: Small constant for numerical stability
        
    Returns:
        Orthogonalized weight matrix
    """
    out_features, in_features = weight.shape
    
    # If the matrix is tall, we can't directly orthogonalize
    if out_features > in_features:
        # Orthogonalize the transpose, then transpose back
        weight_t = weight.t()
        orthogonal_weight_t = orthogonalize(weight_t, eps)
        return orthogonal_weight_t.t()
    
    # Apply Gram-Schmidt process
    for i in range(out_features):
        # Normalize the current row
        weight[i] = F.normalize(weight[i], dim=0, eps=eps)
        
        # Make all subsequent rows orthogonal to this one
        for j in range(i + 1, out_features):
            # Subtract projection
            weight[j] = weight[j] - torch.sum(weight[i] * weight[j]) * weight[i]
    
    return weight


class OrthogonalLinear(nn.Module):
    """
    Linear layer with orthogonal weight matrix.
    
    This layer enforces orthogonality constraints on the weight matrix
    either through regularization or direct orthogonalization.
    """
    
    def __init__(self, in_features, out_features, bias=True, enforce_mode='regularize'):
        """
        Initialize orthogonal linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            enforce_mode: How to enforce orthogonality:
                - 'strict': Apply Gram-Schmidt in forward pass
                - 'regularize': Add regularization term to loss
                - 'init': Only initialize orthogonally
        """
        super(OrthogonalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.enforce_mode = enforce_mode
        
        # Initialize weights orthogonally
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.orthogonal_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """Forward pass with optional orthogonalization."""
        if self.enforce_mode == 'strict':
            # Orthogonalize weights before each forward pass
            with torch.no_grad():
                self.weight.copy_(orthogonalize(self.weight.clone()))
        
        return F.linear(x, self.weight, self.bias)
    
    def get_orthogonality_penalty(self):
        """
        Calculate the orthogonality penalty for regularization.
        
        Returns a measure of how non-orthogonal the weight matrix is.
        Specifically, calculates ||W*W^T - I||_F for the weight matrix W.
        """
        if self.enforce_mode != 'regularize':
            return 0.0
            
        # Calculate W*W^T
        wt = self.weight.transpose(0, 1)
        wwt = torch.matmul(self.weight, wt)
        
        # Get identity matrix of appropriate size
        I = torch.eye(self.weight.size(0), device=self.weight.device)
        
        # Calculate Frobenius norm of the difference
        penalty = torch.norm(wwt - I, p='fro')
        return penalty


class OSPAMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Orthogonal Subspace Projection.
    
    This module implements attention where each head operates in an orthogonal
    subspace of the input representation.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, enforce_mode='regularize',
                 causal=False, device=None, dtype=None):
        """
        Initialize OSPA multi-head attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: If set to False, linear layers will not learn an additive bias
            enforce_mode: How to enforce orthogonality
            causal: Whether to apply causal mask for auto-regressive generation
            device: Device for computation
            dtype: Data type for computation
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OSPAMultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.scaling = self.head_dim ** -0.5
        
        # Create projection parameters using orthogonal layers
        self.q_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, enforce_mode=enforce_mode, **factory_kwargs)
        self.k_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, enforce_mode=enforce_mode, **factory_kwargs)
        self.v_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, enforce_mode=enforce_mode, **factory_kwargs)
        self.o_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, enforce_mode=enforce_mode, **factory_kwargs)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Forward pass for OSPA multi-head attention.
        
        Args:
            query: Query embeddings of shape (N, L, E) where L is the target sequence length,
                  N is the batch size, E is the embedding dimension.
            key: Key embeddings of shape (N, S, E) where S is the source sequence length.
            value: Value embeddings of shape (N, S, E).
            attn_mask: Optional attention mask, broadcastable to (N, L, S).
            key_padding_mask: Optional mask of shape (N, S) indicating which elements
                within key should be ignored for the attention computation.
                
        Returns:
            attn_output: Attention output of shape (N, L, E).
            attn_weights: Attention weights of shape (N, num_heads, L, S).
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Linear projections with orthogonal weights
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape and transpose for multi-head attention
        # [N, L, E] -> [N, L, h, d] -> [N, h, L, d]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale query
        q = q * self.scaling
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [N, h, L, S]
        
        # Apply masks if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # [N, h, L, d]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.embed_dim
        )
        
        # Final projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights
    
    def get_orthogonality_penalty(self):
        """Get the total orthogonality penalty for all projection matrices."""
        penalty = (
            self.q_proj.get_orthogonality_penalty() +
            self.k_proj.get_orthogonality_penalty() +
            self.v_proj.get_orthogonality_penalty() +
            self.o_proj.get_orthogonality_penalty()
        )
        return penalty


class OSPAEncoderLayer(nn.Module):
    """Transformer encoder layer with OSPA attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", enforce_mode='regularize', norm_first=False,
                 layer_norm_eps=1e-5, device=None, dtype=None):
        """
        Initialize OSPA encoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
            enforce_mode: How to enforce orthogonality
            norm_first: If True, use pre-norm architecture
            layer_norm_eps: Layer norm epsilon
            device: Device for computation
            dtype: Data type for computation
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OSPAEncoderLayer, self).__init__()
        
        # Self-attention with OSPA
        self.self_attn = OSPAMultiHeadAttention(
            d_model, nhead, dropout=dropout, enforce_mode=enforce_mode, **factory_kwargs
        )
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = _get_activation_fn(activation)
        
        # Architecture type
        self.norm_first = norm_first
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for encoder layer with OSPA.
        
        Args:
            src: Input sequence of shape (batch_size, seq_len, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Key padding mask
            
        Returns:
            Output sequence and attention weights
        """
        x = src
        
        if self.norm_first:
            # Pre-norm architecture
            attn_output, attn_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + attn_output
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm architecture
            attn_output, attn_weights = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self._ff_block(x))
        
        return x, attn_weights
    
    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x, attn_weights = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        return self.dropout1(x), attn_weights
    
    def _ff_block(self, x):
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def get_orthogonality_penalty(self):
        """Get orthogonality penalty for the layer."""
        return self.self_attn.get_orthogonality_penalty()


class OSPADecoderLayer(nn.Module):
    """Transformer decoder layer with OSPA attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", enforce_mode='regularize', norm_first=False,
                 layer_norm_eps=1e-5, device=None, dtype=None):
        """
        Initialize OSPA decoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
            enforce_mode: How to enforce orthogonality
            norm_first: If True, use pre-norm architecture
            layer_norm_eps: Layer norm epsilon
            device: Device for computation
            dtype: Data type for computation
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OSPADecoderLayer, self).__init__()
        
        # Self-attention with OSPA (causal)
        self.self_attn = OSPAMultiHeadAttention(
            d_model, nhead, dropout=dropout, enforce_mode=enforce_mode, 
            causal=True, **factory_kwargs
        )
        
        # Cross-attention with OSPA
        self.cross_attn = OSPAMultiHeadAttention(
            d_model, nhead, dropout=dropout, enforce_mode=enforce_mode,
            **factory_kwargs
        )
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = _get_activation_fn(activation)
        
        # Architecture type
        self.norm_first = norm_first
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for decoder layer with OSPA.
        
        Args:
            tgt: Target sequence of shape (batch_size, tgt_len, d_model)
            memory: Memory from encoder of shape (batch_size, src_len, d_model)
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Output sequence and attention weights for self and cross attention
        """
        x = tgt
        
        if self.norm_first:
            # Pre-norm architecture
            self_attn_out, self_attn_weights = self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self_attn_out
            
            cross_attn_out, cross_attn_weights = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + cross_attn_out
            
            x = x + self._ff_block(self.norm3(x))
        else:
            # Post-norm architecture
            self_attn_out, self_attn_weights = self._sa_block(
                x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + self_attn_out)
            
            cross_attn_out, cross_attn_weights = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + cross_attn_out)
            
            x = self.norm3(x + self._ff_block(x))
        
        return x, (self_attn_weights, cross_attn_weights)
    
    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x, attn_weights = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        return self.dropout1(x), attn_weights
    
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        """Multi-head attention block for cross-attention."""
        x, attn_weights = self.cross_attn(
            query=x, key=mem, value=mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        return self.dropout2(x), attn_weights
    
    def _ff_block(self, x):
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def get_orthogonality_penalty(self):
        """Get orthogonality penalty for the layer."""
        return (
            self.self_attn.get_orthogonality_penalty() +
            self.cross_attn.get_orthogonality_penalty()
        )


def _get_activation_fn(activation):
    """Helper to get activation function by name."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"Activation {activation} not supported")