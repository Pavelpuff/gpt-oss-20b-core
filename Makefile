.PHONY: help setup format lint train evaluate clean

PYTHON := uv run python

help:
	@echo "Available commands:"
	@echo "  make setup    - Install project dependencies via uv"
	@echo "  make format   - Auto-format code and fix linting errors with Ruff"
	@echo "  make lint     - Check code for linting and formatting issues"
	@echo "  make train    - Train the GPT-OSS-20B architecture"
	@echo "  make evaluate - Run autoregressive generation with Nucleus Sampling"
	@echo "  make clean    - Remove cache, wandb logs, and temporary files"

setup:
	uv sync --all-extras

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

train:
	@echo "[*] Starting training pipeline..."
	$(PYTHON) -m src.train --epochs 8 --batch_size 128

evaluate:
	@echo "[*] Running generation..."
	$(PYTHON) -m src.evaluate --top_p 0.5 --temperature 1.5 --gen_length 10

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache wandb/ logs/
	find . -type d -name "__pycache__" -exec rm -rf {} +