"""
Inference and generation utilities for GPT-OSS-20B.
"""

import argparse
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F

from src.models.transformer import Transformer


@torch.no_grad()
def token_choice_greedy(model_logits: torch.Tensor) -> torch.Tensor:
    """Greedy decoding: selects the most likely token."""
    return torch.argmax(model_logits[:, -1:, :], dim=-1)


@torch.no_grad()
def get_dist_after_with_temp_and_topp(
    model_logits: torch.Tensor, top_p: float, t: float
) -> torch.Tensor:
    """Applies temperature scaling and top-p (nucleus) sampling."""
    scaled_logits = model_logits / 1e-8 if t == 0 else model_logits / t
    probs = F.softmax(scaled_logits, dim=-1)

    if top_p == 1.0:
        return probs
    if top_p == 0.0:
        max_index = torch.argmax(probs, dim=-1, keepdim=True)
        return torch.zeros_like(probs).scatter_(-1, max_index, 1.0)

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    probs = probs.masked_fill(indices_to_remove, 0.0)

    return probs / probs.sum(dim=-1, keepdim=True)


@torch.no_grad()
def token_choice_adv(model_logits: torch.Tensor, top_p: float, t: float) -> torch.Tensor:
    """Selects next token using temperature and nucleus sampling."""
    probs = get_dist_after_with_temp_and_topp(model_logits[:, -1:, :], top_p=top_p, t=t)
    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample()


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_seq: torch.Tensor,
    gen_length: int,
    device: torch.device,
    token_choice: Callable[[torch.Tensor], torch.Tensor] = token_choice_greedy,
) -> torch.Tensor:
    """Autoregressive text generation."""
    model.eval()
    current_seq = input_seq.to(device)
    output_tokens = []

    for _ in range(gen_length):
        logits = model(current_seq)
        next_token = token_choice(logits)
        output_tokens.append(next_token)
        current_seq = torch.cat([current_seq, next_token], dim=-1)

    return torch.cat(output_tokens, dim=-1)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[*] Using hardware accelerator: {device}")

    model = Transformer(
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        use_sliding_window_alternating=args.use_swa,
        use_moe=args.use_moe,
    ).to(device)

    seq_len = 64
    input_data = torch.ones((1, seq_len - 10), dtype=torch.long) * 2

    print(f"[*] Initial sequence: {input_data.tolist()[0][:10]}... (length: {input_data.shape[1]})")
    print(
        f"[*] Generating {args.gen_length} tokens with Top-P={args.top_p}, "
        f"Temperature={args.temperature}..."
    )

    if args.top_p == 0.0 and args.temperature == 0.0:
        choice_fn = token_choice_greedy
    else:
        choice_fn = partial(token_choice_adv, top_p=args.top_p, t=args.temperature)

    generated_tokens = generate(
        model=model,
        input_seq=input_data,
        gen_length=args.gen_length,
        device=device,
        token_choice=choice_fn,
    )

    print(f"[*] Generated tokens: {generated_tokens.tolist()[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for GPT-OSS-20B")
    parser.add_argument("--top_p", type=float, default=0.5, help="Nucleus sampling threshold")
    parser.add_argument("--temperature", type=float, default=1.5, help="Softmax temperature")
    parser.add_argument("--gen_length", type=int, default=10, help="Number of tokens to generate")

    parser.add_argument("--vocab_size", type=int, default=7)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--use_swa", action="store_true", help="Enable alternating sliding window")
    parser.add_argument("--use_moe", action="store_true", help="Enable Mixture of Experts")

    main(parser.parse_args())
