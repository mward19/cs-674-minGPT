import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import math

# Source: https://www.stephendiehl.com/posts/post_transformers/#swiglu
class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, dim_out=None, bias=True):
        super().__init__()
        dim_hidden = dim_hidden or 4 * dim_in
        dim_out = dim_out or dim_in
        
        # Linear transformations for gating
        self.w1 = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_hidden, bias=bias)
        
        # Output projection
        self.w3 = nn.Linear(dim_hidden, dim_out, bias=bias)
    
    def forward(self, x):
        # SwiGLU applies SiLU activation to one branch and gates it with the other
        hidden1 = self.w1(x)
        hidden2 = self.w2(x)
        
        # SiLU (Swish) activation: x * sigmoid(x)
        hidden1_act = hidden1 * torch.sigmoid(hidden1)
        
        # Element-wise product for gating
        hidden = hidden1_act * hidden2
        
        # Output projection
        return self.w3(hidden)

# Source: https://www.stephendiehl.com/posts/post_transformers/#rotary-positional-embedding
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, interleaved=False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        
        # Generate inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len, device=None):
        # Get device from buffer if not specified
        if device is None:
            device = self.inv_freq.device
            
        # Generate position indices
        positions = torch.arange(seq_len, device=device).float()
        
        # Compute sinusoidal patterns
        freqs = torch.outer(positions, self.inv_freq)
        
        # Get sine and cosine embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb)[:, :self.dim]
        sin = torch.sin(emb)[:, :self.dim]
        
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, interleaved=False):
    # Apply rotary embeddings to queries and keys
    batch_size, num_heads, seq_len, head_dim = q.shape
    cos = cos.reshape(1, 1, seq_len, cos.shape[-1])  # [1, 1, seq_len, dim/2]
    sin = sin.reshape(1, 1, seq_len, sin.shape[-1])  # [1, 1, seq_len, dim/2]
    
    # Split queries and keys for rotation
    half_dim = head_dim // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    
    # Apply rotation using half-dim rotary embeddings
    q_rotated = torch.cat([
        q1 * cos - q2 * sin,
        q2 * cos + q1 * sin
    ], dim=-1)
    
    k_rotated = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin
    ], dim=-1)
    
    return q_rotated, k_rotated


# Source: https://www.stephendiehl.com/posts/post_transformers/#learning-rate-warmup
class LinearWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # During warmup: linearly increase from 0 to base LR
            scale = float(self.last_epoch + 1) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # After warmup: use base learning rate
            return self.base_lrs


# Source: https://www.stephendiehl.com/posts/post_transformers/#cosine-schedule
class CosineAnnealingWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # During warmup: linearly increase from 0 to base LR
            scale = float(self.last_epoch + 1) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # After warmup: cosine decay from base LR to min_lr
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            # Cosine decay formula: min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
            return [base_lr * scale for base_lr in self.base_lrs]
    
# Source: https://www.stephendiehl.com/posts/post_transformers/#rmsnorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x):
        # Calculate root mean square along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize by RMS
        x_normalized = x / rms
        
        # Apply scaling if using learnable parameters
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
        
        return x_normalized