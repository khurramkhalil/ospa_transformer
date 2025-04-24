import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import orthogonal_


class OrthogonalLinear(nn.Module):
    """
    Linear layer that enforces orthogonality constraints on its weight matrix.
    
    Modes:
    - 'init': Initialize weights orthogonally but don't enforce during training
    - 'regularize': Add a regularization term to the loss to encourage orthogonality
    - 'strict': Enforce orthogonality by projection after each update
    """
    
    def __init__(self, in_features, out_features, bias=True, mode='init', eps=1e-8):
        super(OrthogonalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.eps = eps
        
        # Handle case when out_features > in_features for orthogonal initialization
        if out_features > in_features:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.is_transposed = True
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.is_transposed = False
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.is_transposed:
            # For out_features > in_features, initialize the transpose orthogonally
            # and then transpose back
            temp_weight = torch.zeros(self.in_features, self.out_features)
            orthogonal_(temp_weight)
            self.weight.data = temp_weight.t()
        else:
            orthogonal_(self.weight)
            
        if self.bias is not None:
            # Initialize bias to zero following PyTorch's default
            nn.init.zeros_(self.bias)
            
    def forward(self, x):
        # If in strict mode, project weight to orthogonal matrix before forward pass
        if self.mode == 'strict':
            self._enforce_orthogonality()
            
        return F.linear(x, self.weight, self.bias)
    
    def _enforce_orthogonality(self):
        """Enforce orthogonality by projecting to the nearest orthogonal matrix."""
        with torch.no_grad():
            if self.is_transposed:
                # For out_features > in_features, operate on W^T
                u, s, v = torch.svd(self.weight.t(), some=False)
                self.weight.data = (u @ v.t()).t()
            else:
                u, s, v = torch.svd(self.weight, some=False)
                self.weight.data = u @ v.t()
    
    def compute_orthogonality_penalty(self):
        """Compute the Frobenius norm of the difference between W^T W and I."""
        if self.is_transposed:
            # For out_features > in_features, compute ||W W^T - I||_F
            prod = self.weight @ self.weight.t()
            identity = torch.eye(self.out_features, device=self.weight.device)
        else:
            # For out_features <= in_features, compute ||W^T W - I||_F
            prod = self.weight.t() @ self.weight
            identity = torch.eye(self.in_features, device=self.weight.device)
            
        return torch.norm(prod - identity, p='fro')