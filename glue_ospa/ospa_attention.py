# ospa_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging # Add logging
from orthogonal_linear import OrthogonalLinear

logger = logging.getLogger(__name__) # Setup logger for this module

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
            
    # def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
    #             attn_mask=None, average_attn_weights=True):
    #     """
    #     Input shape: Time x Batch x Channel (T x B x C)
    #     or Batch x Time x Channel (B x T x C) if batch_first=True
    #     """
    #     # Default shapes are L x B x D (Length x Batch x Dimension)
    #     tgt_len, bsz, embed_dim = query.size()
    #     src_len = key.size(0)
        
    #     # Apply linear projections to get queries, keys, and values
    #     q = self.q_proj(query)
    #     k = self.k_proj(key)
    #     v = self.v_proj(value)
        
    #     # Reshape q, k, v for multi-head attention
    #     # [L, B, D] -> [L, B, H, D/H] -> [B, H, L, D/H]
    #     q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
    #     k = k.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
    #     v = v.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

    #     # If specified, add zero attention
    #     if self.add_zero_attn:
    #         zero_attn_shape = (bsz, self.num_heads, 1, self.head_dim)
    #         k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=2)
    #         v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=2)
    #         if attn_mask is not None:
    #             attn_mask = torch.cat([
    #                 attn_mask, 
    #                 torch.zeros(attn_mask.size(0), 1, dtype=attn_mask.dtype, device=attn_mask.device)
    #             ], dim=1)
    #         if key_padding_mask is not None:
    #             key_padding_mask = torch.cat([
    #                 key_padding_mask, 
    #                 torch.zeros(key_padding_mask.size(0), 1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
    #             ], dim=1)
                
    #     # Calculate attention scores
    #     # [B, H, L, D/H] x [B, H, D/H, S] = [B, H, L, S]
    #     scaling = float(self.head_dim) ** -0.5
    #     q = q * scaling
    #     attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
    #     # Apply attention mask if provided
    #     if attn_mask is not None:
    #         if attn_mask.dim() == 2:
    #             attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [S, S] -> [1, 1, S, S]
    #         elif attn_mask.dim() == 3:
    #             attn_mask = attn_mask.unsqueeze(1)  # [B, S, S] -> [B, 1, S, S]
    #         attn_weights = attn_weights + attn_mask
            
    #     # Apply key padding mask if provided
    #     if key_padding_mask is not None:
    #         attn_weights = attn_weights.masked_fill(
    #             key_padding_mask.unsqueeze(1).unsqueeze(2),
    #             float('-inf')
    #         )
            
    #     # Apply softmax and dropout
    #     attn_weights = F.softmax(attn_weights, dim=-1)
    #     attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
    #     # Apply attention weights to values
    #     # [B, H, L, S] x [B, H, S, D/H] = [B, H, L, D/H]
    #     attn_output = torch.matmul(attn_weights, v)
        
    #     # Reshape output and apply output projection
    #     # [B, H, L, D/H] -> [L, B, D]
    #     attn_output = attn_output.transpose(1, 2).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    #     attn_output = self.out_proj(attn_output)
        
    #     if need_weights:
    #         if average_attn_weights:
    #             # Average attention weights over heads
    #             attn_weights = attn_weights.mean(dim=1)
    #         return attn_output, attn_weights
    #     else:
    #         return attn_output, None


    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, batch_idx_for_debug=-1): # Add batch_idx_for_debug
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        # --- 1. Projections ---
        q_proj_out = self.q_proj(query)
        k_proj_out = self.k_proj(key)
        v_proj_out = self.v_proj(value)

        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): q_proj_out - min={q_proj_out.min():.2e}, max={q_proj_out.max():.2e}, has_nan={torch.isnan(q_proj_out).any()}")
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): k_proj_out - min={k_proj_out.min():.2e}, max={k_proj_out.max():.2e}, has_nan={torch.isnan(k_proj_out).any()}")
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): v_proj_out - min={v_proj_out.min():.2e}, max={v_proj_out.max():.2e}, has_nan={torch.isnan(v_proj_out).any()}")
        if torch.isnan(q_proj_out).any() or torch.isnan(k_proj_out).any() or torch.isnan(v_proj_out).any():
            logger.error(f"!!! NAN after QKV orthogonal projections (Batch {batch_idx_for_debug}) !!!")
            # Print norms of projection weights if OSPA
            if hasattr(self.q_proj, 'weight'): logger.error(f"  q_proj.weight norm: {self.q_proj.weight.norm().item():.2e}")
            if hasattr(self.k_proj, 'weight'): logger.error(f"  k_proj.weight norm: {self.k_proj.weight.norm().item():.2e}")
            if hasattr(self.v_proj, 'weight'): logger.error(f"  v_proj.weight norm: {self.v_proj.weight.norm().item():.2e}")
            raise ValueError("NaN from OSPA QKV projections")


        # --- 2. Reshape and Scaling ---
        q_reshaped = q_proj_out.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k_reshaped = k_proj_out.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v_reshaped = v_proj_out.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        scaling = float(self.head_dim) ** -0.5
        q_scaled = q_reshaped * scaling
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): q_scaled (after scaling) - min={q_scaled.min():.2e}, max={q_scaled.max():.2e}, has_nan={torch.isnan(q_scaled).any()}")
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): k_reshaped - min={k_reshaped.min():.2e}, max={k_reshaped.max():.2e}, has_nan={torch.isnan(k_reshaped).any()}")
        if torch.isnan(q_scaled).any() or torch.isnan(k_reshaped).any():
            logger.error(f"!!! NAN after q scaling or in k_reshaped (Batch {batch_idx_for_debug}) !!!")
            raise ValueError("NaN from q_scaled or k_reshaped")

        # --- 3. Attention Scores (QK^T) ---
        # attn_weights_raw shape: [bsz, num_heads, tgt_len, src_len]
        attn_weights_raw = torch.matmul(q_scaled, k_reshaped.transpose(-2, -1))
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): attn_weights_raw (QK^T) - min={attn_weights_raw.min():.2e}, max={attn_weights_raw.max():.2e}, has_nan={torch.isnan(attn_weights_raw).any()}")
        if torch.isnan(attn_weights_raw).any():
            logger.error(f"!!! NAN in raw attention scores (QK^T) (Batch {batch_idx_for_debug}) !!!")
            # This is a very likely spot. If q_scaled or k_reshaped have very large values, matmul can overflow.
            raise ValueError("NaN from QK^T matmul")


        # --- 4. Apply attn_mask (Causal or other) ---
        attn_weights_masked = attn_weights_raw # Start with raw
        if attn_mask is not None:
            # Ensure attn_mask is broadcastable
            # attn_mask might be [tgt_len, src_len] or [bsz*num_heads, tgt_len, src_len]
            # PyTorch's MHA handles broadcasting. Let's assume it's correctly shaped or broadastable.
            # For causal mask, it's usually [tgt_len, src_len] with -inf
            # For padding mask, it's applied differently (see below)
            # We need to align with how nn.MultiheadAttention expects it if adapting from there.
            # Or use the logic you had before for reshaping.
            # Current PyTorch nn.MHA expects attn_mask to be [L,S] or [N*H, L, S] or [L, N*H, S]
            # If causal mask is [L,S]
            if attn_mask.dim() == 2 and attn_mask.shape == (tgt_len, src_len):
                 attn_weights_masked = attn_weights_raw + attn_mask.unsqueeze(0).unsqueeze(0) # Add to all batches and heads
            elif attn_mask.dim() == 3 and attn_mask.shape[0] == bsz * self.num_heads: # For [N*H, L, S]
                 attn_weights_masked = attn_weights_raw + attn_mask.unsqueeze(1) # This might be wrong dim for add
            # Add other conditions if your mask comes in different shapes
            else:
                 logger.warning(f"DEBUG MHA (Batch {batch_idx_for_debug}): Unexpected attn_mask shape {attn_mask.shape}, not applying directly by addition. Check if it's a boolean mask for key_padding_mask.")
            
            logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): attn_weights (after attn_mask) - min={attn_weights_masked.min():.2e}, max={attn_weights_masked.max():.2e}, has_nan={torch.isnan(attn_weights_masked).any()}")
            if torch.isnan(attn_weights_masked).any():
                logger.error(f"!!! NAN after applying attn_mask (Batch {batch_idx_for_debug}) !!!")
                raise ValueError("NaN from attn_mask application")
        
        current_attn_scores = attn_weights_masked

        # --- 5. Apply key_padding_mask ---
        # key_padding_mask shape: [bsz, src_len], True for PAD
        if key_padding_mask is not None:
            # Expand to [bsz, 1, 1, src_len] for broadcasting over heads and query positions
            current_attn_scores = current_attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), # [B, 1, 1, S]
                float('-inf')
            )
            logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): attn_weights (after key_padding_mask) - min={current_attn_scores.min():.2e}, max={current_attn_scores.max():.2e}, has_nan={torch.isnan(current_attn_scores).any()}")
            if torch.isnan(current_attn_scores).any():
                logger.error(f"!!! NAN after applying key_padding_mask (Batch {batch_idx_for_debug}) !!!")
                raise ValueError("NaN from key_padding_mask application")

        # ... before softmax ...
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): Input to softmax (current_attn_scores) - min={current_attn_scores.min():.2e}, max={current_attn_scores.max():.2e}, has_nan={torch.isnan(current_attn_scores).any()}")

        # Check if any row is all -inf
        # current_attn_scores shape: [bsz, num_heads, tgt_len, src_len]
        is_all_neg_inf = torch.all(current_attn_scores == float('-inf'), dim=-1) # Check along the key/src_len dimension
        if torch.any(is_all_neg_inf):
            logger.error(f"!!! AT LEAST ONE ROW IN INPUT TO SOFTMAX IS ALL -INF (Batch {batch_idx_for_debug}) !!!")
            # Find which batch items/heads/query_pos have this issue
            problematic_indices = (is_all_neg_inf == True).nonzero(as_tuple=False)
            logger.error(f"  Problematic indices (batch_in_global_attn_matrix, head, query_pos_in_tgt_len): {problematic_indices.tolist()}")
            # You might want to inspect the original input_ids and masks for these problematic indices
            # For example, print key_padding_mask for the problematic batch items:
            unique_problem_batch_indices = problematic_indices[:,0].unique()
            logger.error(f"  Unique batch indices in current mini-batch with all -inf rows: {unique_problem_batch_indices.tolist()}")

            if key_padding_mask is not None:
                 for b_idx_in_minibatch in unique_problem_batch_indices:
                     logger.error(f"  key_padding_mask for batch item {b_idx_in_minibatch.item()} (within this minibatch): {key_padding_mask[b_idx_in_minibatch.item()]}")
                #  for b_idx in problematic_indices[:,0].unique(): # Iterate through unique batch indices with problems
                #      logger.error(f"  key_padding_mask for batch item {b_idx.item()}: {key_padding_mask[b_idx.item()]}")
            if attn_mask is not None:
                logger.error(f"  Content of attn_mask (shape {attn_mask.shape}): {attn_mask}")
            else:
                logger.error(f"  attn_mask was None.")
            raise ValueError("Softmax input has all -inf row")

            # if attn_mask is not None: # Causal mask for LM, typically
            #      # attn_mask might be [tgt_len, src_len]
            #      # For a problematic query_pos, inspect its row in attn_mask
            #      for b, h, q_pos in problematic_indices.tolist():
            #          logger.error(f"  attn_mask row for query_pos {q_pos} (if applicable): {attn_mask[q_pos] if attn_mask.dim()==2 and q_pos < attn_mask.shape[0] else 'Mask not 2D or q_pos out of bounds'}")
            # This is where you'd raise the error or handle it if you have a specific strategy
            # raise ValueError("NaN from softmax due to all -inf row") # Keep this for now

        # --- 6. Softmax ---
        attn_weights_softmax = F.softmax(current_attn_scores, dim=-1)
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): attn_weights (after softmax) - min={attn_weights_softmax.min():.2e}, max={attn_weights_softmax.max():.2e}, has_nan={torch.isnan(attn_weights_softmax).any()}")
        # Check if softmax outputted NaNs (e.g., if all inputs were -inf)
        if torch.isnan(attn_weights_softmax).any():
            logger.error(f"!!! NAN in attention weights AFTER SOFTMAX (Batch {batch_idx_for_debug}) !!!")
            logger.error(f"  Input to softmax (current_attn_scores) - min={current_attn_scores.min():.2e}, max={current_attn_scores.max():.2e}, has_nan={torch.isnan(current_attn_scores).any()}")
            raise ValueError("NaN from softmax")


        # --- 7. Dropout on Attention Weights ---
        attn_weights_dropout = F.dropout(attn_weights_softmax, p=self.dropout_p, training=self.training)
        # logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): attn_weights (after dropout) - min={attn_weights_dropout.min():.2e}, max={attn_weights_dropout.max():.2e}, has_nan={torch.isnan(attn_weights_dropout).any()}")
        if torch.isnan(attn_weights_dropout).any():
             logger.error(f"!!! NAN after attention dropout (Batch {batch_idx_for_debug}) !!!")
             raise ValueError("NaN from attention dropout")


        # --- 8. Apply to Values ---
        # v_reshaped shape: [bsz, num_heads, src_len, head_dim]
        # attn_weights_dropout shape: [bsz, num_heads, tgt_len, src_len]
        attn_output = torch.matmul(attn_weights_dropout, v_reshaped)
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): attn_output (after matmul with V) - min={attn_output.min():.2e}, max={attn_output.max():.2e}, has_nan={torch.isnan(attn_output).any()}")
        if torch.isnan(attn_output).any():
            logger.error(f"!!! NAN in attention output (after matmul with V) (Batch {batch_idx_for_debug}) !!!")
            logger.error(f"  v_reshaped - min={v_reshaped.min():.2e}, max={v_reshaped.max():.2e}, has_nan={torch.isnan(v_reshaped).any()}")
            logger.error(f"  attn_weights_dropout - min={attn_weights_dropout.min():.2e}, max={attn_weights_dropout.max():.2e}, has_nan={torch.isnan(attn_weights_dropout).any()}")
            raise ValueError("NaN from matmul(attn_weights, V)")


        # --- 9. Reshape and Output Projection ---
        attn_output_reshaped = attn_output.transpose(1, 2).contiguous().view(tgt_len, bsz, embed_dim)
        final_attn_output = self.out_proj(attn_output_reshaped)
        logger.info(f"DEBUG MHA (Batch {batch_idx_for_debug}): final_attn_output (after out_proj) - min={final_attn_output.min():.2e}, max={final_attn_output.max():.2e}, has_nan={torch.isnan(final_attn_output).any()}")
        if torch.isnan(final_attn_output).any():
            logger.error(f"!!! NAN after final output projection (Batch {batch_idx_for_debug}) !!!")
            logger.error(f"  out_proj.weight norm: {self.out_proj.weight.norm().item():.2e}")
            raise ValueError("NaN from out_proj")

        # Return weights for debugging from the encoder layer
        # The encoder layer was: src2_attn_output, attn_weights = self.self_attn(...)
        # So we need to return attn_weights_softmax (or attn_weights_dropout) as the second element
        returned_attn_weights = attn_weights_softmax if not average_attn_weights else attn_weights_softmax.mean(dim=1)

        if need_weights:
            return final_attn_output, returned_attn_weights
        else:
            return final_attn_output, None

    def get_orthogonality_penalty(self):
        """Calculate the total orthogonality penalty for all projection matrices."""
        penalty = 0.0
        if self.orth_mode == 'regularize':
            penalty += self.q_proj.compute_orthogonality_penalty()
            penalty += self.k_proj.compute_orthogonality_penalty()
            penalty += self.v_proj.compute_orthogonality_penalty()
            penalty += self.out_proj.compute_orthogonality_penalty()
        return penalty