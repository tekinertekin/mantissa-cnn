"""Accuracy benchmark — SKELETON. Same protocol as bench.speed (2 archs x
5 datasets x stratified 2000/1000 subsets, 3 epochs, batch 32, lr 0.01,
seed 0); this module reports test accuracy instead of time.

For every (architecture, dataset, contender): fit on the train subset,
report train and test accuracy plus the final training loss. Contenders and
fairness rules are speed.py's; sklearn's MLPClassifier on flattened pixels
stays explicitly labeled as the non-CNN baseline. No number goes in the
README that this script did not produce.

Output: bench/results/accuracy.json
  {"env": {...}, "protocol": {...},
   "accuracy": {"<arch>/<dataset>": {"<contender>":
                {"train_acc": ..., "test_acc": ..., "final_loss": ...}}}}

Run from the repo root:  python -m bench.accuracy
"""
from __future__ import annotations

from .speed import (ARCHITECTURES, BATCH_SIZE, CONTENDERS, DATASETS, EPOCHS,
                    LR, N_TEST, N_TRAIN, SEED)

__all__ = ["ARCHITECTURES", "DATASETS", "CONTENDERS", "N_TRAIN", "N_TEST",
           "EPOCHS", "BATCH_SIZE", "LR", "SEED"]


def main() -> int:
    raise NotImplementedError("bench skeleton — implementation lands in the "
                              "benchmark phase; the docstring is the spec")


if __name__ == "__main__":
    raise SystemExit(main())
