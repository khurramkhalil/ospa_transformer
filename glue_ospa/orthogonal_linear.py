# orthogonal_linear.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_

class OrthogonalLinear(nn.Module):
    """
    Linear layer that enforces orthogonality constraints on its weight matrix.
    Modes:
    - 'init': Orthogonal initialization only
    - 'regularize': Adds a soft penalty to loss
    - 'strict': Projects weights to the nearest orthogonal matrix every N steps
    """
    def __init__(self, in_features, out_features, bias=True, mode='init', eps=1e-8, orthogonalize_every_n_steps=1):
        super(OrthogonalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.eps = eps
        self.orthogonalize_every_n_steps = orthogonalize_every_n_steps
        self.step_count = 0
        # Handle transposition for orthogonal projection when needed
        self.is_transposed = out_features > in_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.is_transposed:
            temp_weight = torch.zeros(self.in_features, self.out_features)
            orthogonal_(temp_weight)
            self.weight.data = temp_weight.t()
        else:
            orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x):
        if self.mode == 'strict':
            self.step_count += 1
            if self.step_count % self.orthogonalize_every_n_steps == 0:
                self._enforce_orthogonality()
        return F.linear(x, self.weight, self.bias)
        
    def _enforce_orthogonality(self):
        """Project weight matrix to the nearest orthogonal matrix using SVD."""
        with torch.no_grad():
            if self.is_transposed:
                W = self.weight.t()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                self.weight.data = (U @ Vh).t()
            else:
                U, S, Vh = torch.linalg.svd(self.weight, full_matrices=False)
                self.weight.data = U @ Vh
                
    def compute_orthogonality_penalty(self):
        """Compute ||W^T W - I||_F or ||W W^T - I||_F depending on shape."""
        if self.is_transposed:
            prod = self.weight @ self.weight.t()
            identity = torch.eye(self.out_features, device=self.weight.device)
        else:
            prod = self.weight.t() @ self.weight
            identity = torch.eye(self.in_features, device=self.weight.device)
        return torch.norm(prod - identity, p='fro').pow(2)