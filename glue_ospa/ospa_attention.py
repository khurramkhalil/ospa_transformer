import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from orthogonal_linear import OrthogonalLinear


class OSPAMultiHeadAttention(nn.Module):
    """
    Orthogonal Subspace Projection Attention (OSPA) module.
    Each attention head operates in an orthogonal subspace ensuring disentangled representations.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, 
                 orth_mode='regularize', add_bias_kv=False, add_zero_attn=False):
        super(OSPAMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.orth_mode = orth_mode
        
        # Orthogonal projections for Q, K, V
        self.q_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, mode=orth_mode)
        self.k_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, mode=orth_mode)
        self.v_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, mode=orth_mode)
        
        # Output projection
        self.out_proj = OrthogonalLinear(embed_dim, embed_dim, bias=bias, mode=orth_mode)
        
        self.add_zero_attn = add_zero_attn
        
        # Optional bias for key and value (not common but included for compatibility)
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
    
    def get_orthogonality_penalty(self):
        """Calculate the total orthogonality penalty for all projection matrices."""
        penalty = 0.0
        if self.orth_mode == 'regularize':
            penalty += self.q_proj.compute_orthogonality_penalty()
            penalty += self.k_proj.compute_orthogonality_penalty()
            penalty += self.v_proj.compute_orthogonality_penalty()
            penalty += self.out_proj.compute_orthogonality_penalty()
        return penalty