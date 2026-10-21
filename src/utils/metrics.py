"""
Evaluation metrics and logging utilities.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb


@torch.no_grad()
def eval_acc(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Calculates overall accuracy across the dataset."""
    model.eval()
    sum_acc = 0.0
    num_examples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model_out = model(x)
        acc = (torch.argmax(model_out, dim=-1) == y).float().sum().item()
        sum_acc += acc
        num_examples += model_out.shape[0] * model_out.shape[1]

    return sum_acc / num_examples


@torch.no_grad()
def calc_per_token_acc(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Calculates accuracy per position in the sequence."""
    model.eval()
    correct_counts = None
    total_counts = None

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=-1)
        corrected_mask = pred == y

        if correct_counts is None:
            seq_len = x.shape[1]
            correct_counts = torch.zeros(seq_len, device=device, dtype=torch.float32)
            total_counts = torch.zeros(seq_len, device=device, dtype=torch.float32)

        correct_counts += corrected_mask.float().sum(dim=0)
        total_counts += torch.tensor(x.shape[0], device=device, dtype=torch.float32)

    return (correct_counts / total_counts).cpu().numpy()


def eval_fn(
    epoch: int, model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> None:
    """Evaluation routine hooked into the training loop."""
    acc = eval_acc(model, dataloader, device)
    print(f"Epoch {epoch}: Avg eval accuracy {acc:.4f}")
    if wandb.run is not None:
        wandb.log({"eval/accuracy": acc, "epoch": epoch})
