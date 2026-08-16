PYTHON ?= python

.PHONY: install format lint test smoke generate-small train-small eval-small

install:
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m black src scripts tests experiments
	$(PYTHON) -m ruff check --fix src scripts tests experiments

lint:
	$(PYTHON) -m ruff check src scripts tests experiments
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests -q

smoke: test generate-small train-small eval-small

generate-small:
	$(PYTHON) scripts/generate_dataset.py --config configs/data_surface_code.yaml --out data/processed/small_dataset.npz

train-small:
	$(PYTHON) scripts/train_predecoder.py --config configs/train_predecoder.yaml --data data/processed/small_dataset.npz --out checkpoints/tiny_predecoder.pt

eval-small:
	$(PYTHON) scripts/evaluate_realtime.py --config configs/eval_realtime.yaml --data data/processed/small_dataset.npz --checkpoint checkpoints/tiny_predecoder.pt --out results/runs/smoke_eval
