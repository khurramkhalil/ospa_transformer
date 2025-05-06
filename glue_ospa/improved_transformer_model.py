# improved_transformer_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Add logging

# Assuming ospa_transformer.py exists and defines OSPATransformer
try:
    from ospa_transformer import OSPATransformer
except ImportError:
     print("Warning: ospa_transformer.py not found. OSPA type will not work.")
     OSPATransformer = None # Define as None if not found

# Assuming baseline_models.py defines VanillaTransformer and LinformerTransformer
try:
    from baseline_models import VanillaTransformer, LinformerTransformer # Example import
except ImportError:
     print("Warning: baseline_models.py not found. Vanilla/Linformer types will not work.")
     VanillaTransformer = None
     LinformerTransformer = None

logger = logging.getLogger(__name__) # Setup logger for this module


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # Shape [max_len, 1, d_model] for broadcasting
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Keep pe shape as [max_len, 1, d_model]. Transpose in forward if needed.
        # Or register as [1, max_len, d_model] if using batch_first=True elsewhere
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add positional encoding to input embeddings
        # self.pe is [max_len, 1, d_model], x is [seq_len, batch_size, d_model]
        # Select required length and add using broadcasting (batch dim is broadcast)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Container Transformer model that handles embeddings, positional encoding,
    different underlying Transformer backbones (Vanilla, OSPA), and task-specific heads,
    designed to work with Hugging Face standard inputs (input_ids, attention_mask).
    """

    def __init__(self, transformer_type: str, vocab_size: int, d_model: int, nhead: int,
                 nlayers: int, num_labels: int, # Added num_labels
                 dropout: float = 0.1, dim_feedforward: int = 2048,
                 orth_mode: str = "regularize", orth_penalty_weight: float = 0.01,
                 task: str = "lm", activation: str = "relu", **kwargs): # Allow kwargs
        super().__init__()
        # --- Store essential config ---
        self.transformer_type = transformer_type
        self.task = task
        self.vocab_size = vocab_size # Store vocab size
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.num_labels = num_labels # Store num_labels for classification head
        self.orth_mode = orth_mode
        # orth_penalty_weight is used when calculating loss in train script, not directly here
        logger.info(f"Initializing TransformerModel: type={transformer_type}, task={task}, vocab={vocab_size}, d_model={d_model}, nhead={nhead}, nlayers={nlayers}, num_labels={num_labels}, orth_mode={orth_mode}")


        # --- Input Embeddings & Positional Encoding ---
        self.token_encoder = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Scaling factor for embeddings (common practice)
        self.embedding_scale = math.sqrt(self.d_model)

        # --- Underlying Transformer Backbone ---
        if transformer_type == "ospa":
            if OSPATransformer is None: raise ImportError("OSPATransformer not available.")
            # OSPATransformer expects an encoder-decoder structure, but we only use encoder for LM/GLUE
            self.transformer = OSPATransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0, # No decoder needed for these tasks
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                orth_mode=orth_mode
                # orth_penalty_weight is handled externally via get_orthogonality_penalty
            )
            # We will only call self.transformer.encoder in forward
        elif transformer_type == "vanilla":
            if VanillaTransformer is None:
                 # Use PyTorch standard if baseline not found
                 logger.warning("VanillaTransformer not found, using nn.TransformerEncoder.")
                 encoder_layer = nn.TransformerEncoderLayer(
                     d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                     dropout=dropout, activation=activation, batch_first=False # Assuming seq_len first
                 )
                 encoder_norm = nn.LayerNorm(d_model)
                 self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, norm=encoder_norm)
                 self.transformer = None # Mark that we use transformer_encoder directly
            else:
                 # Use custom VanillaTransformer if available
                 self.transformer = VanillaTransformer(
                     d_model=d_model, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=0,
                     dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
                 )
        # Add Linformer or other types similarly if needed
        # elif transformer_type == "linformer": ...
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")

        # --- Task-Specific Output Head ---
        # Dropout before the final layer is common
        self.output_dropout = nn.Dropout(dropout)
        if task == "lm":
            # Output layer maps final hidden states to vocabulary logits
            self.output_layer = nn.Linear(d_model, self.vocab_size)
            # Optional: Weight tying (share weights between input embedding and output layer)
            # self.output_layer.weight = self.token_encoder.weight
        elif task == "glue" or task == "classification":
            # Output layer maps the [CLS] token representation (or avg pooling) to class logits/regression value
            self.output_layer = nn.Linear(d_model, self.num_labels)
        else:
            raise ValueError(f"Unsupported task for output head: {task}")

        # --- Initialization ---
        self.init_weights()


    def init_weights(self):
        """Initialize model weights."""
        initrange = 0.02 # A common range for Transformer embeddings/layers
        # Initialize input embeddings
        nn.init.uniform_(self.token_encoder.weight, -initrange, initrange)

        # Initialize output layer (important for LM stability)
        if hasattr(self, 'output_layer'):
            nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1) # Small gain can help LM
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)
        # Note: Underlying Transformer layers (Vanilla/OSPA) might have their own init


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """
        Forward pass.

        Args:
            input_ids: Tensor of token IDs, shape [seq_len, batch_size]
            attention_mask: Tensor indicating padding from HF Tokenizer, shape [seq_len, batch_size] (1 for real, 0 for pad)
                           OR [batch_size, seq_len] if coming directly from tokenizer without transpose.
                           This will be converted to src_key_padding_mask.
            **kwargs: Catches other potential inputs like token_type_ids.
        """
        # Input_ids is expected to be [SeqLen, BatchSize]
        seq_len, batch_size = input_ids.shape
        print(f"Batch {kwargs.get('batch_idx_for_debug', -1)} - src_pos: min={src_pos.min().item():.2e}, max={src_pos.max().item():.2e}, mean={src_pos.mean().item():.2e}, has_nan={torch.isnan(src_pos).any()}")
        # --- 1. Embeddings and Positional Encoding ---
        embeds = self.token_encoder(input_ids) * self.embedding_scale
        src_pos = self.pos_encoder(embeds) # Output: [SeqLen, BatchSize, Dim]

        if torch.isnan(src_pos).any():
            logger.error("NaN detected after embedding/positional encoding!")
            raise ValueError("NaN input to transformer layers")

        # --- 2. Prepare Masks ---
        causal_src_mask = None # For causal LM mask
        src_key_padding_mask = None # For padding

        # Create causal mask for Language Modeling task
        if self.task == 'lm':
             try:
                 causal_src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
             except AttributeError:
                 logger.warning("nn.Transformer.generate_square_subsequent_mask not found. Implement manually if PyTorch < 1.9.")
                 # Manual fallback:
                 mask = (torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device)) == 1).transpose(0, 1)
                 causal_src_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


        # Create PyTorch-style padding mask from Hugging Face attention_mask
        # HF attention_mask: 1 for non-pad, 0 for pad.
        # PyTorch src_key_padding_mask: True for pad, False for non-pad. Shape: [BatchSize, SeqLen]
        if attention_mask is not None:
            # First, ensure attention_mask is [BatchSize, SeqLen] if it came as [SeqLen, BatchSize]
            if attention_mask.shape == (seq_len, batch_size):
                attention_mask_bs_first = attention_mask.transpose(0, 1) # Convert to [BatchSize, SeqLen]
            elif attention_mask.shape == (batch_size, seq_len):
                attention_mask_bs_first = attention_mask
            else:
                logger.error(f"Unexpected attention_mask shape: {attention_mask.shape}. Expected ({seq_len}, {batch_size}) or ({batch_size}, {seq_len}). Cannot create padding mask.")
                attention_mask_bs_first = None # Cannot proceed safely

            if attention_mask_bs_first is not None:
                src_key_padding_mask = (attention_mask_bs_first == 0) # True where attention_mask is 0 (pad)
        else:
            # If no attention_mask is provided, assume no padding.
            # This might happen for LM if all sequences in a block are full.
            # Or, if the tokenizer didn't return one and all tokens are considered valid.
            pass # src_key_padding_mask remains None

        # --- 3. Pass through Transformer Encoder ---
        encoder_module = None
        if self.transformer_type == "ospa":
             if self.transformer and hasattr(self.transformer, 'encoder'):
                 encoder_module = self.transformer.encoder
        elif self.transformer_type == "vanilla":
             if hasattr(self, 'transformer_encoder'):
                 encoder_module = self.transformer_encoder
             elif self.transformer and hasattr(self.transformer, 'encoder'):
                 encoder_module = self.transformer.encoder

        if encoder_module is None:
             raise RuntimeError(f"Could not find valid encoder module for transformer_type '{self.transformer_type}'")

        transformer_output = encoder_module(
            src=src_pos,                      # [SeqLen, BatchSize, Dim]
            mask=causal_src_mask,             # [SeqLen, SeqLen] (for LM) or None
            src_key_padding_mask=src_key_padding_mask  # [BatchSize, SeqLen] (True for PAD)
        )
        # Output shape: [seq_len, batch_size, d_model]
        print(f"Batch {kwargs.get('batch_idx_for_debug', -1)} - transformer_output: min={transformer_output.min().item():.2e}, max={transformer_output.max().item():.2e}, mean={transformer_output.mean().item():.2e}, has_nan={torch.isnan(transformer_output).any()}")

        if torch.isnan(transformer_output).any():
            logger.error("NaN detected after transformer encoder layers!")
            raise ValueError("NaN output from transformer layers")

        # --- 4. Task-Specific Head ---
        output_features = self.output_dropout(transformer_output)

        if self.task == "lm":
            final_output = self.output_layer(output_features) # [SeqLen, BatchSize, VocabSize]
        elif self.task == "glue" or self.task == "classification":
            cls_representation = output_features[0, :, :] # [BatchSize, Dim] (representation of first token)
            final_output = self.output_layer(cls_representation) # [BatchSize, NumLabels]
        else:
             raise ValueError(f"Task head not implemented for task: {self.task}")

        return final_output


    def get_orthogonality_penalty(self):
        """
        Calculates orthogonality penalty ONLY from the encoder component,
        as the decoder is not used for LM/GLUE tasks in this setup.
        """
        if self.transformer_type == "ospa":
            # Access the encoder component directly within the instantiated OSPATransformer
            if self.transformer and hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'get_orthogonality_penalty'):
                return self.transformer.encoder.get_orthogonality_penalty()
            else:
                 logger.warning("Could not find OSPA encoder with get_orthogonality_penalty method.")
                 return 0.0 # Return 0 if method not found
        return 0.0 # Return 0 for non-OSPA models


    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resizes the input token embeddings matrix and the output LM head
        to accommodate a new vocabulary size (e.g., after adding tokens).
        """
        old_num_tokens, old_embedding_dim = self.token_encoder.weight.size()
        if old_num_tokens == new_num_tokens:
            logger.info("Token embeddings size already matches new vocab size. No resize needed.")
            return

        # Create new embedding layer
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        # Initialize new embeddings (e.g., Xavier uniform)
        nn.init.normal_(new_embeddings.weight, std=0.02) # Common init for new tokens

        # Copy weights from the old matrix
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.token_encoder.weight.data[:num_tokens_to_copy, :]

        self.token_encoder = new_embeddings
        self.vocab_size = new_num_tokens # Update stored vocab size

        # If task is LM, resize the output layer too
        if self.task == 'lm' and hasattr(self, 'output_layer'):
            old_output_tokens, old_output_dim = self.output_layer.weight.size()
            # Bias shape check
            old_output_bias_size = self.output_layer.bias.size(0) if self.output_layer.bias is not None else 0


            if old_output_dim != old_embedding_dim:
                 logger.warning(f"Output layer input dim {old_output_dim} differs from embedding dim {old_embedding_dim}. Check model consistency.")

            new_output_layer = nn.Linear(old_output_dim, new_num_tokens)
            nn.init.xavier_uniform_(new_output_layer.weight, gain=0.1) # Re-initialize
            if new_output_layer.bias is not None: nn.init.zeros_(new_output_layer.bias)

            # Copy existing weights and biases
            num_output_tokens_to_copy = min(old_output_tokens, new_num_tokens)
            new_output_layer.weight.data[:num_output_tokens_to_copy, :] = self.output_layer.weight.data[:num_output_tokens_to_copy, :]
            if self.output_layer.bias is not None and new_output_layer.bias is not None:
                num_output_bias_to_copy = min(old_output_bias_size, new_num_tokens)
                new_output_layer.bias.data[:num_output_bias_to_copy] = self.output_layer.bias.data[:num_output_bias_to_copy]


            self.output_layer = new_output_layer
            logger.info(f"Resized token embeddings and LM output layer from {old_num_tokens} to {new_num_tokens}.")
        else:
             logger.info(f"Resized token embeddings from {old_num_tokens} to {new_num_tokens}.")