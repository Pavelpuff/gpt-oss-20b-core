"""
Main training pipeline for the GPT-OSS-20B model.
"""

import argparse
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.data.generator import example_generator, generate_dataset, generate_recursive
from src.models.transformer import Transformer
from src.utils.metrics import eval_fn


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    eval_func: partial,
    num_epochs: int,
    device: torch.device,
) -> None:
    """Standard training loop with W&B logging."""
    for epoch in range(num_epochs):
        if epoch == 0:
            eval_func(epoch, model)

        model.train()
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item()})

        eval_func(epoch + 1, model)


def main(args: argparse.Namespace) -> None:
    wandb.init(project="gpt-oss-20b", config=vars(args))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[*] Using hardware accelerator: {device}")

    # Dataset Preparation
    perm_gen = example_generator(
        lambda: generate_recursive(args.initial, args.vocab_size, args.next_prob)
    )

    test_dataset, generators = generate_dataset(perm_gen, args.seq_len, 1000)
    train_dataset, _ = generate_dataset(perm_gen, args.seq_len, 10000, exclude=generators)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model Initialization
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("[*] Starting training loop...")
    train(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        eval_func=partial(eval_fn, dataloader=test_loader, device=device),
        num_epochs=args.epochs,
        device=device,
    )

    wandb.finish()
    print("[*] Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-OSS-20B")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=7)
    parser.add_argument("--initial", type=int, default=2)
    parser.add_argument("--next_prob", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)

    # Model architecture args
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--use_swa", action="store_true", help="Enable alternating sliding window")
    parser.add_argument("--use_moe", action="store_true", help="Enable Mixture of Experts")

    main(parser.parse_args())
