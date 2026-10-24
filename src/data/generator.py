"""
Data generation module for synthetic sequence modeling.
"""
import torch
import numpy as np
from typing import Callable, Set, Tuple
from torch.utils.data import TensorDataset


def generate_recursive(n_first: int, vocab_size: int, next_prob: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Generates initial parameters for a recursive sequence."""
    assert 0 < vocab_size
    initial = np.random.randint(0, vocab_size, n_first)
    coeffs = np.random.randint(0, vocab_size, n_first)
    return initial, coeffs, vocab_size, next_prob


class SeqGen:
    """Generates recurrent sequences with stochastically repeating terms."""

    def __init__(self, initial: np.ndarray, coeffs: np.ndarray, size: int, next_prob: float) -> None:
        assert len(coeffs) == len(initial)
        self.initial = initial
        self.coeffs = coeffs
        self.size = size
        self.next_prob = next_prob
        self.current = initial

    def __iter__(self) -> 'SeqGen':
        return self

    def __next__(self) -> int:
        if np.random.random() < self.next_prob:
            new = self.current[-1] + 1
        else:
            new = (self.current @ self.coeffs)

        new %= self.size
        self.current = np.append(self.current, new)[1:]
        return int(new)

    def __key(self) -> tuple:
        return (tuple(self.initial), tuple(self.coeffs), self.size, self.next_prob)

    def __hash__(self) -> int:
        return hash(self.__key())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SeqGen):
            return self.__key() == other.__key()
        return False


def generate_dataset(
        gen_factory: Callable[[], 'SeqGen'],
        seq_len: int,
        num_entries: int,
        exclude: list['SeqGen'] | None = None
) -> Tuple[TensorDataset, Set['SeqGen']]:
    """
    Generates a TensorDataset consisting of input sequences and next-token targets.
    """
    exclude_set = set(exclude) if exclude is not None else set()
    entries = []
    generators = set()

    for _ in range(num_entries):
        while True:
            seq_gen = gen_factory()
            if seq_gen in exclude_set:
                continue

            seq = [next(seq_gen) for _ in range(seq_len + 1)]
            break

        generators.add(seq_gen)
        entries.append(seq)

    data = torch.tensor(entries, dtype=torch.long)
    x = data[:, :seq_len]
    y = data[:, 1:]

    return TensorDataset(x, y), generators


def example_generator(gen: Callable) -> Callable[[], SeqGen]:
    """Factory wrapper for SeqGen initialization."""

    def example_gen() -> SeqGen:
        return SeqGen(*gen())

    return example_gen