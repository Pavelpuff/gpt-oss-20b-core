"""
Core architectural layers for the GPT-OSS-20B Transformer model.
Includes RoPE, GQA, Sliding Window Attention, and MoE blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """Feed-Forward network using SwiGLU activation."""

    def __init__(self, hidden_dim: int, inner_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.v = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.w1(x)
        hv = self.v(x)
        h2 = F.silu(h1) * hv
        return self.w2(h2)


class RotaryPositionalEmbedding(nn.Module):
    """Pairwise Rotary Positional Embedding (RoPE)."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int) -> None:
        thetas = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        position_ids = torch.arange(seq_len).float()
        pos_thetas = torch.outer(position_ids, thetas)
        self.cos_cached = pos_thetas.cos()
        self.sin_cached = pos_thetas.sin()

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        curr_sin = self.sin_cached[start_pos : start_pos + seq_len].to(x.device)
        curr_cos = self.cos_cached[start_pos : start_pos + seq_len].to(x.device)

        curr_sin = curr_sin.view(1, 1, seq_len, head_dim // 2, 1)
        curr_cos = curr_cos.view(1, 1, seq_len, head_dim // 2, 1)

        x_reshaped = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x_out = torch.zeros_like(x_reshaped)

        x_out[..., 0] = x_reshaped[..., 0] * curr_cos.squeeze(-1) - x_reshaped[
            ..., 1
        ] * curr_sin.squeeze(-1)
        x_out[..., 1] = x_reshaped[..., 0] * curr_sin.squeeze(-1) + x_reshaped[
            ..., 1
        ] * curr_cos.squeeze(-1)

        return x_out.flatten(-2)


def calculate_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_weights: torch.Tensor,
    rope: RotaryPositionalEmbedding,
    scale: float,
    device: torch.device,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes scaled dot-product attention with RoPE and GQA handling."""
    q, k = rope(q), rope(k)

    if q.size(1) != k.size(1):
        num_queries_per_kv = q.size(1) // k.size(1)
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)

    k = k * key_weights.view(1, -1, 1, 1)
    attn_scores = (q @ k.transpose(-1, -2)) * scale

    if mask is None:
        seq_len = q.size(2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = torch.where(mask, 0.0, -float("inf"))

    attn_probs = F.softmax(attn_scores + mask[None, None, ...], dim=-1)
    return attn_probs @ v


def calculate_sliding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_weights: torch.Tensor,
    rope: RotaryPositionalEmbedding,
    scale: float,
    device: torch.device,
    window_size: int,
) -> torch.Tensor:
    """Computes sliding window attention."""
    seq_len = q.size(2)
    q, k = rope(q), rope(k)
    k = k * key_weights.view(1, -1, 1, 1)

    attn_scores = (q @ k.transpose(-1, -2)) * scale

    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    mask = torch.where(mask, 0.0, -float("inf"))

    second_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=-(window_size + 1)
    )
    mask += torch.where(second_mask, -float("inf"), 0.0)

    attn_probs = F.softmax(attn_scores + mask[None, None, ...], dim=-1)
    return attn_probs @ v


class GroupedQueryAttention(nn.Module):
    """Multi-Head Grouped Query Attention module."""

    def __init__(
        self, hidden_dim: int, num_heads: int, head_dim: int, num_kv_heads: int | None = None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.key_weights = nn.Parameter(torch.randn(num_heads))
        self.rope = RotaryPositionalEmbedding(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        output = calculate_attention(q, k, v, self.key_weights, self.rope, self.scale, x.device)
        return self.out_proj(output.transpose(1, 2).flatten(-2))


class SWAttention(nn.Module):
    """Sliding Window Attention module."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, window_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.key_weights = nn.Parameter(torch.randn(num_heads))
        self.rope = RotaryPositionalEmbedding(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        output = calculate_sliding_attention(
            q, k, v, self.key_weights, self.rope, self.scale, x.device, self.window_size
        )
        return self.out_proj(output.transpose(1, 2).flatten(-2))


class Router(nn.Module):
    """Top-K Router for Mixture of Experts."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.weights = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.weights(x)
        top_k_values, expert_indices = torch.topk(x, self.top_k, dim=-1, largest=True)
        return F.softmax(top_k_values, dim=-1), expert_indices


class MixtureOfExperts(nn.Module):
    """Mixture of Experts (MoE) layer replacing standard FFN."""

    def __init__(
        self, hidden_dim: int, inner_dim: int, num_experts: int = 8, top_k: int = 2
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.router = Router(hidden_dim, num_experts, top_k)
        self.ffns = nn.ModuleList(
            [SwiGLUFeedForward(hidden_dim, inner_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        routing_weights, expert_indices = self.router(x)

        flat_x = x.view(-1, hidden_dim)
        flat_weights = routing_weights.view(-1, self.top_k)
        flat_indices = expert_indices.view(-1, self.top_k)
        output = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.ffns):
            mask_indices = flat_indices == i
            tokens_mask = mask_indices.any(dim=-1)

            if tokens_mask.any():
                selected_input = flat_x[tokens_mask]
                expert_out = expert(selected_input.unsqueeze(1)).squeeze(1)
                weight = flat_weights[mask_indices]
                output[tokens_mask] += expert_out * weight.unsqueeze(-1)

        return output.view(batch, seq_len, hidden_dim)
