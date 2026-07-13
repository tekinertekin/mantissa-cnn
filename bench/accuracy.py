"""Accuracy benchmark. Same protocol as bench.speed (2 archs x 5 datasets x
stratified 2000/1000 subsets, 3 epochs, batch 32, lr 0.01, seed 0); this
module reports test accuracy instead of time.

For every (architecture, dataset, contender): fit on the train subset,
report train and test accuracy plus the final training loss. Contenders and
fairness rules are speed.py's (scikit-learn is absent by design — it has no
convolutional layers). No number goes in the README that this script did
not produce.

Output: bench/results/accuracy.json
  {"env": {...}, "protocol": {...},
   "accuracy": {"<arch>/<dataset>": {"<contender>":
                {"train_acc": ..., "test_acc": ..., "final_loss": ...}}}}

Run from the repo root:  python -m bench.accuracy
"""
from __future__ import annotations

import json
import time

import numpy as np

from .speed import (ARCHITECTURES, BATCH_SIZE, CONTENDERS, DATASETS, EPOCHS,
                    LR, N_TEST, N_TRAIN, REPO_ROOT, SEED, _contenders,
                    _env_block, _load_pair, _native)

__all__ = ["ARCHITECTURES", "DATASETS", "CONTENDERS", "N_TRAIN", "N_TEST",
           "EPOCHS", "BATCH_SIZE", "LR", "SEED"]

RESULTS_PATH = REPO_ROOT / "bench" / "results" / "accuracy.json"


def _eval_pair(arch, dataset, contenders):
    """One fit per contender under the fixed budget; returns
    {contender: {train_acc, test_acc, final_loss}}."""
    Xtr, ytr, Xte, yte = _load_pair(dataset)
    in_shape = tuple(Xtr.shape[1:])
    native = _native(contenders, Xtr, ytr, Xte, yte)
    out = {}
    for name, factory, _prep in contenders:
        Xn, yn, Xtn = native[name]
        est = factory(arch, in_shape).fit(Xn, yn)
        out[name] = {
            "train_acc": float(np.mean(np.asarray(est.predict(Xn)) == ytr)),
            "test_acc": float(np.mean(np.asarray(est.predict(Xtn)) == yte)),
            "final_loss": est.final_loss_,
        }
    return out


def main() -> int:
    contenders = _contenders()
    names = [n for n, *_ in contenders]
    print(f"contenders: {', '.join(names)}")

    accuracy = {}
    t_start = time.perf_counter()
    for arch in ARCHITECTURES:
        for dataset in DATASETS:
            pair = f"{arch}/{dataset}"
            print(f"\n[{pair}] fit + evaluate ...")
            accuracy[pair] = _eval_pair(arch, dataset, contenders)
            for name in names:
                r = accuracy[pair][name]
                print(f"  {name:12s} train {r['train_acc']:.3f}  "
                      f"test {r['test_acc']:.3f}  loss {r['final_loss']:.4f}")

    out = {
        "env": _env_block(),
        "protocol": {"architectures": list(ARCHITECTURES),
                     "datasets": list(DATASETS),
                     "n_train": N_TRAIN, "n_test": N_TEST,
                     "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                     "seed": SEED},
        "accuracy": accuracy,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)} "
          f"({time.perf_counter() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
