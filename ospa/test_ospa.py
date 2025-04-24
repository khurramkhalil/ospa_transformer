import torch
import unittest
from orthogonal_linear import OrthogonalLinear
from ospa_attention import OSPAMultiHeadAttention
from ospa_transformer import OSPATransformerEncoderLayer, OSPATransformer


class TestOrthogonalLinear(unittest.TestCase):
    """Tests for the OrthogonalLinear layer."""
    
    def test_initialization(self):
        """Test if weights are initialized orthogonally."""
        for in_features, out_features in [(64, 64), (128, 64), (64, 128)]:
            layer = OrthogonalLinear(in_features, out_features)
            if out_features <= in_features:
                        # Forward pass
        output, attn_weights = attn(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (seq_len, batch_size, d_model))
        
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (batch_size, seq_len, seq_len))
        
    def test_orthogonality_penalty(self):
        """Test orthogonality penalty calculation."""
        d_model, num_heads = 64, 4
        
        # Initialize with regularize mode
        attn = OSPAMultiHeadAttention(d_model, num_heads, orth_mode='regularize')
        
        # Initial penalty should be small since weights are initialized orthogonally
        penalty = attn.get_orthogonality_penalty()
        self.assertGreaterEqual(penalty, 0.0, "Penalty should be non-negative")
        
        # Make weights non-orthogonal
        with torch.no_grad():
            attn.q_proj.weight.data = torch.randn_like(attn.q_proj.weight)
            attn.k_proj.weight.data = torch.randn_like(attn.k_proj.weight)
            attn.v_proj.weight.data = torch.randn_like(attn.v_proj.weight)
            attn.out_proj.weight.data = torch.randn_like(attn.out_proj.weight)
        
        # Penalty should be larger now
        new_penalty = attn.get_orthogonality_penalty()
        self.assertGreater(new_penalty, penalty, "Penalty should increase after making weights non-orthogonal")


class TestOSPATransformer(unittest.TestCase):
    """Tests for the OSPATransformer components."""
    
    def test_encoder_layer(self):
        """Test OSPATransformerEncoderLayer forward pass."""
        d_model, nhead = 64, 4
        batch_size, seq_len = 2, 10
        
        # Create encoder layer
        layer = OSPATransformerEncoderLayer(d_model, nhead)
        
        # Create input
        src = torch.randn(seq_len, batch_size, d_model)
        
        # Forward pass
        output = layer(src)
        
        # Check output shape
        self.assertEqual(output.shape, src.shape)
        
    def test_transformer_forward(self):
        """Test full OSPATransformer model."""
        d_model, nhead = 64, 4
        batch_size, src_len, tgt_len = 2, 10, 8
        
        # Create transformer
        transformer = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128
        ).to(device='cpu')  # Explicitly set device to CPU for testing
        
        # Create source and target tensors
        src = torch.randn(src_len, batch_size, d_model)
        tgt = torch.randn(tgt_len, batch_size, d_model)
        
        # Create causal mask for decoder
        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len) * float('-inf'),
            diagonal=1
        )
        
        # Forward pass
        output = transformer(src, tgt, tgt_mask=tgt_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (tgt_len, batch_size, d_model))
        
    def test_orthogonality_penalty_propagation(self):
        """Test that orthogonality penalties propagate through the model."""
        transformer = OSPATransformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            orth_mode='regularize'
        ).to(device='cpu')  # Explicitly set device to CPU for testing
        
        # Initial penalty
        penalty = transformer.get_orthogonality_penalty()
        self.assertGreaterEqual(penalty, 0.0, "Penalty should be non-negative")
        
        # Make some weights non-orthogonal
        with torch.no_grad():
            for p in transformer.parameters():
                if p.dim() > 1 and p.shape[0] == p.shape[1]:
                    p.data = torch.randn_like(p)
        
        # New penalty should be larger
        new_penalty = transformer.get_orthogonality_penalty()
        self.assertGreater(new_penalty, penalty, "Penalty should increase after making weights non-orthogonal")


# if __name__ == "__main__":
#     unittest.main() 
#     Test W^T W ≈ I
#                 prod = layer.weight.t() @ layer.weight
#                 identity = torch.eye(in_features)
#                 diff = torch.norm(prod - identity, p='fro')
#                 self.assertLess(diff, 1e-5, f"Orthogonality error: {diff} for shape {layer.weight.shape}")
#             else:
#                 # Test W W^T ≈ I
#                 prod = layer.weight @ layer.weight.t()
#                 identity = torch.eye(out_features)
#                 diff = torch.norm(prod - identity, p='fro')
#                 self.assertLess(diff, 1e-5, f"Orthogonality error: {diff} for shape {layer.weight.shape}")
    
    def test_strict_mode(self):
        """Test if strict mode enforces orthogonality after weight updates."""
        layer = OrthogonalLinear(64, 64, mode='strict')
        
        # Create random input
        x = torch.randn(10, 64)
        
        # Force weights to be non-orthogonal
        with torch.no_grad():
            layer.weight.data = torch.randn_like(layer.weight)
        
        # Forward pass should enforce orthogonality
        _ = layer(x)
        
        # Check orthogonality after forward pass
        prod = layer.weight.t() @ layer.weight
        identity = torch.eye(64)
        diff = torch.norm(prod - identity, p='fro')
        self.assertLess(diff, 1e-5, f"Orthogonality error after strict update: {diff}")
        
    def test_regularize_mode(self):
        """Test if regularize mode computes the correct penalty."""
        layer = OrthogonalLinear(64, 64, mode='regularize')
        
        # Initialize as orthogonal should give near-zero penalty
        penalty1 = layer.compute_orthogonality_penalty()
        self.assertLess(penalty1, 1e-5, f"Initial penalty should be near zero: {penalty1}")
        
        # Make weights non-orthogonal
        with torch.no_grad():
            layer.weight.data = torch.randn_like(layer.weight)
        
        # Penalty should be larger now
        penalty2 = layer.compute_orthogonality_penalty()
        self.assertGreater(penalty2, 1.0, f"Penalty after randomization should be larger: {penalty2}")


class TestOSPAAttention(unittest.TestCase):
    """Tests for the OSPAMultiHeadAttention module."""
    
    def test_forward_shape(self):
        """Test if attention outputs have correct shapes."""
        batch_size, seq_len, d_model = 2, 10, 64
        num_heads = 4
        
        # Create model and inputs
        attn = OSPAMultiHeadAttention(d_model, num_heads)
        query = torch.randn(seq_len, batch_size, d_model)
        key = torch.randn(seq_len, batch_size, d_model)
        value = torch.randn(seq_len, batch_size, d_model)
        
        #