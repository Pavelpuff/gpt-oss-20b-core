"""
High-level Transformer model assembling structural blocks.
"""

import torch
import torch.nn as nn

from src.models.layers import (
    GroupedQueryAttention,
    MixtureOfExperts,
    SWAttention,
    SwiGLUFeedForward,
)


class TransformerBlock(nn.Module):
    """A single Transformer Decoder block with optional MoE and Sliding Window."""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        head_dim: int,
        use_sliding_window: bool = False,
        window_size: int = 128,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_dim)
        self.post_attention_norm = nn.RMSNorm(hidden_dim)

        if use_sliding_window:
            self.attention = SWAttention(hidden_dim, num_heads, head_dim, window_size)
        else:
            self.attention = GroupedQueryAttention(hidden_dim, num_heads, head_dim, num_kv_heads)

        if use_moe:
            self.ffn = MixtureOfExperts(hidden_dim, ff_dim, num_experts, top_k)
        else:
            self.ffn = SwiGLUFeedForward(hidden_dim, ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x))
        x = x + self.ffn(self.post_attention_norm(x))
        return x


class Transformer(nn.Module):
    """Full GPT-OSS-20B architecture."""

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        head_dim: int,
        use_sliding_window_alternating: bool = False,
        window_size: int = 128,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    ff_dim=ff_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    use_sliding_window=(use_sliding_window_alternating and (i % 2 == 1)),
                    window_size=window_size,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_kv_heads=num_kv_heads,
                )
                for i in range(n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(self.final_norm(x))
