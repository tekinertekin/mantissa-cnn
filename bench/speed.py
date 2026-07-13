"""Speed + peak-RSS benchmark vs famous CNN implementations.

Protocol (fixed; do not tune per contender):
  - Architectures: lenet5 and minivgg from mantissa_cnn.models, identical
    layer shapes re-expressed in each framework.
  - Datasets: all five (mnist, fashion_mnist, kmnist, cifar10, qmnist),
    stratified subsets via datasets.subset(name, N_TRAIN, N_TEST, SEED).
  - Training: EPOCHS epochs, batch BATCH_SIZE, plain SGD lr=LR, seed SEED.
  - Repeats: REPEATS, INTERLEAVED round-robin (A,B,C,... xR) so thermal and
    background drift hit every contender equally; report medians, keep raw
    samples in the JSON. time.perf_counter(); fit() wall time only (data
    loaded and framework imported beforehand).
  - Batch predict over the test subset: median of PREDICT_CALLS calls.
  - PEAK RSS: one (contender, arch, dataset) per fresh subprocess; child
    imports its own framework, fits once, reports
    resource.getrusage(RUSAGE_SELF).ru_maxrss (BYTES on macOS, KiB on
    Linux — normalize). Import cost deliberately included: users pay it.

Contenders:
  ours        mantissa_cnn (backend="mantissa"; the C engine)
  ours_numpy  mantissa_cnn (backend="numpy"; the reference backend)
  torch       torch.nn.Sequential with the same layers/init family, SGD
  tensorflow  tf.keras.Sequential, same layers, SGD (compile once; exclude
              tracing from timing, as with any one-time JIT)
  sklearn     MLPClassifier on FLATTENED pixels — sklearn cannot express a
              CNN; this is an explicitly-labeled non-CNN baseline, present
              to show what giving up convolutions costs, not as a rival.

Fairness: identical epochs/batch/lr/seed everywhere the API allows; record
library versions, CPU and thread settings in the JSON. Measure, don't assume.

Output: bench/results/speed.json
  {"env": {...}, "protocol": {...},
   "fit_s":      {"<arch>/<dataset>": {"<contender>": {"median": ..., "samples": [...]}}},
   "predict_ms": {...same nesting...},
   "peak_rss_mb": {"<arch>/<dataset>": {"<contender>": ...}}}

Run from the repo root:  python -m bench.speed
(the RSS worker re-invokes:  python -m bench.speed --worker <contender> <arch> <dataset>)
"""
from __future__ import annotations

import json
import os
import platform

# Keep TensorFlow's C++ banner out of benchmark output (set before any TF
# import anywhere in the process).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

# numpy 2.x on Apple Accelerate emits spurious FPE RuntimeWarnings from the
# BLAS matmul kernel even on finite inputs (verified: contender weights stay
# bounded). They fire from both our numpy backend and sklearn's internals.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

# --- protocol (fixed) --------------------------------------------------------
ARCHITECTURES = ("lenet5", "minivgg")
DATASETS = ("mnist", "fashion_mnist", "kmnist", "cifar10", "qmnist")
N_TRAIN = 2000
N_TEST = 1000
EPOCHS = 3
BATCH_SIZE = 32
LR = 0.01
SEED = 0
REPEATS = 5
PREDICT_CALLS = 20
CONTENDERS = ("ours", "ours_numpy", "torch", "tensorflow", "sklearn")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "bench" / "results" / "speed.json"

# Samples for the one untimed warm-up fit per contender before its timed
# repeats: first-call runtime setup (TF graph machinery spin-up, torch kernel
# dispatch caches, our engine's dylib load) is a one-time JIT-like cost,
# excluded the same way imports are. Applied uniformly to every contender.
WARMUP_N = 64


# --- contender estimators ----------------------------------------------------
# Common shape: factory(arch, in_shape) -> fresh estimator (weights
# initialized — construction is untimed; fit() is the timed region);
# fit(X, y) trains under the fixed protocol and sets final_loss_;
# predict(X) -> integer class ids. Heavy imports live inside the classes so
# an RSS worker only pays for the framework it actually uses.

class _OursCNN:
    """mantissa_cnn.models.<arch> on the chosen backend."""

    def __init__(self, arch, in_shape, backend):
        from mantissa_cnn import models
        self._net = getattr(models, arch)(in_shape=in_shape, seed=SEED,
                                          backend=backend)

    def fit(self, X, y):
        self._net.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
        self.final_loss_ = float(self._net.history_["loss"][-1])
        return self

    def predict(self, X):
        return self._net.predict(X)


def _torch_model(arch, in_shape):
    """The exact lenet5/minivgg layer shapes as torch.nn.Sequential.
    Spatial sizes mirror mantissa_cnn.layers' floor semantics (torch's
    Conv2d/MaxPool2d use the same floor formula)."""
    import torch.nn as nn
    c, h, w = in_shape
    if arch == "lenet5":
        fh, fw = (h // 2 - 4) // 2, (w // 2 - 4) // 2   # pad-2 conv, pool, valid conv, pool
        seq = [nn.Conv2d(c, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
               nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
               nn.Linear(16 * fh * fw, 120), nn.ReLU(),
               nn.Linear(120, 84), nn.ReLU(),
               nn.Linear(84, 10)]
    elif arch == "minivgg":
        fh, fw = h // 4, w // 4                          # two same-pad blocks, two pools
        seq = [nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
               nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
               nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
               nn.Flatten(),
               nn.Linear(64 * fh * fw, 128), nn.ReLU(),
               nn.Linear(128, 10)]
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return nn.Sequential(*seq)


class _TorchCNN:
    """Same layers and init family as ours: He normal on relu layers, Glorot
    uniform on the logits layer, zero biases (torch's default init family
    differs and does not train to a useful model in this 3-epoch budget —
    measured; matching the init keeps the comparison about the frameworks,
    not the initializer)."""

    def __init__(self, arch, in_shape):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        self._m = _torch_model(arch, in_shape)
        g = torch.Generator().manual_seed(SEED)
        params = [m for m in self._m.modules()
                  if isinstance(m, (nn.Conv2d, nn.Linear))]
        for m in params:
            nn.init.zeros_(m.bias)
        for m in params[:-1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu", generator=g)
        nn.init.xavier_uniform_(params[-1].weight, generator=g)
        self._rng = np.random.default_rng(SEED)   # seeded epoch shuffle

    def fit(self, X, y):
        import torch
        m = self._m
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.0)
        loss_fn = torch.nn.CrossEntropyLoss()
        n = len(X)
        for _ in range(EPOCHS):
            order = self._rng.permutation(n)
            loss_sum = 0.0
            for s in range(0, n, BATCH_SIZE):
                idx = torch.from_numpy(order[s:s + BATCH_SIZE])
                opt.zero_grad()
                loss = loss_fn(m(X[idx]), y[idx])
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(idx)
            self.final_loss_ = loss_sum / n
        return self

    def predict(self, X):
        import torch
        self._m.eval()
        with torch.no_grad():
            return self._m(X).argmax(1).numpy()


class _KerasCNN:
    """tf.keras.Sequential, same layers and init family as ours ('same'
    padding == pad 2 for 5x5 / pad 1 for 3x3 at stride 1). Built + compiled
    in the constructor — outside the timed region, like any one-time setup;
    each repeat still gets a fresh model, so weight init is per repeat."""

    def __init__(self, arch, in_shape):
        import keras
        keras.utils.set_random_seed(SEED)   # init + fit(shuffle=True) shuffling
        c, h, w = in_shape
        L = keras.layers
        he = keras.initializers.HeNormal(seed=SEED)
        glorot = keras.initializers.GlorotUniform(seed=SEED)
        if arch == "lenet5":
            layers = [
                L.Conv2D(6, 5, padding="same", activation="relu", kernel_initializer=he),
                L.MaxPool2D(2),
                L.Conv2D(16, 5, activation="relu", kernel_initializer=he),
                L.MaxPool2D(2),
                L.Flatten(),
                L.Dense(120, activation="relu", kernel_initializer=he),
                L.Dense(84, activation="relu", kernel_initializer=he),
                L.Dense(10, kernel_initializer=glorot),
            ]
        elif arch == "minivgg":
            layers = [
                L.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer=he),
                L.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer=he),
                L.MaxPool2D(2),
                L.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=he),
                L.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=he),
                L.MaxPool2D(2),
                L.Flatten(),
                L.Dense(128, activation="relu", kernel_initializer=he),
                L.Dense(10, kernel_initializer=glorot),
            ]
        else:
            raise ValueError(f"unknown arch {arch!r}")
        m = keras.Sequential([keras.Input((h, w, c))] + layers)
        m.compile(optimizer=keras.optimizers.SGD(learning_rate=LR, momentum=0.0),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self._m = m

    def fit(self, X, y):
        h = self._m.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        shuffle=True, verbose=0)
        self.final_loss_ = float(h.history["loss"][-1])
        return self

    def predict(self, X):
        return self._m.predict(X, verbose=0).argmax(1)


# The MLP hidden sizes mirror each architecture's dense head (lenet5:
# 120 -> 84, minivgg: 128) — the closest model sklearn can express; it sees
# flattened pixels because it has no convolutions. Non-CNN baseline.
_SK_HIDDEN = {"lenet5": (120, 84), "minivgg": (128,)}


class _SkMLP:
    def __init__(self, arch, in_shape):
        from sklearn.neural_network import MLPClassifier
        self._clf = MLPClassifier(
            hidden_layer_sizes=_SK_HIDDEN[arch], solver="sgd",
            learning_rate_init=LR, batch_size=BATCH_SIZE, max_iter=EPOCHS,
            momentum=0.0, nesterovs_momentum=False, shuffle=True,
            random_state=SEED)

    def fit(self, X, y):
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            # max_iter=EPOCHS is the protocol's budget, not a convergence bug.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._clf.fit(X, y)
        self.final_loss_ = float(self._clf.loss_curve_[-1])
        return self

    def predict(self, X):
        return self._clf.predict(X)


# --- contender registry ------------------------------------------------------
# (name, factory, prep). prep maps the numpy NCHW subset into the contender's
# native form ONCE, outside the timed region, so fit() measures training only.

def _prep_ours(X, y):
    return X, y                       # already contiguous NCHW float32 / int32


def _prep_torch(X, y):
    import torch
    return (torch.from_numpy(X),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)))


def _prep_tf(X, y):
    return np.ascontiguousarray(X.transpose(0, 2, 3, 1)), y   # NHWC


def _prep_sk(X, y):
    return X.reshape(len(X), -1), y   # flattened pixels (non-CNN baseline)


def _contenders():
    reg = [
        ("ours", lambda a, s: _OursCNN(a, s, "mantissa"), _prep_ours),
        ("ours_numpy", lambda a, s: _OursCNN(a, s, "numpy"), _prep_ours),
        ("torch", _TorchCNN, _prep_torch),
        ("tensorflow", _KerasCNN, _prep_tf),
        ("sklearn", _SkMLP, _prep_sk),
    ]
    assert tuple(n for n, *_ in reg) == CONTENDERS
    return reg


# --- data --------------------------------------------------------------------

def _load_pair(dataset):
    from mantissa_cnn import datasets
    return datasets.subset(dataset, N_TRAIN, N_TEST, SEED)


def _native(contenders, Xtr, ytr, Xte, yte):
    """{name: (X_train, y_train, X_test)} in each contender's native form."""
    out = {}
    for name, _factory, prep in contenders:
        Xn, yn = prep(Xtr, ytr)
        Xtn, _ = prep(Xte, yte)
        out[name] = (Xn, yn, Xtn)
    return out


# --- RSS worker --------------------------------------------------------------

def _rss_mb() -> float:
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss: bytes on macOS, KiB on Linux.
    if sys.platform == "darwin":
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def _run_worker(contender: str, arch: str, dataset: str) -> int:
    """Fresh subprocess: import the contender's framework, fit once under the
    full protocol, print peak RSS in MB. Import cost is included on purpose —
    it is what a user pays."""
    spec = {name: (factory, prep) for name, factory, prep in _contenders()
            }.get(contender)
    if spec is None:
        print(f"unknown contender {contender!r}", file=sys.stderr)
        return 2
    factory, prep = spec
    Xtr, ytr, _Xte, _yte = _load_pair(dataset)
    Xn, yn = prep(Xtr, ytr)
    factory(arch, tuple(Xtr.shape[1:])).fit(Xn, yn)
    print(f"{_rss_mb():.4f}")
    return 0


def _measure_rss(contender: str, arch: str, dataset: str) -> float:
    proc = subprocess.run(
        [sys.executable, "-m", "bench.speed", "--worker", contender, arch, dataset],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"RSS worker failed for {contender}/{arch}/{dataset}:\n{proc.stderr}")
    return float(proc.stdout.strip().splitlines()[-1])


# --- timing ------------------------------------------------------------------

def _time_pair(arch, dataset, contenders):
    """Interleaved round-robin timing for one (arch, dataset). Returns
    (fit_s, predict_ms) dicts keyed by contender."""
    Xtr, ytr, Xte, yte = _load_pair(dataset)
    in_shape = tuple(Xtr.shape[1:])
    native = _native(contenders, Xtr, ytr, Xte, yte)

    # One untimed warm-up fit per contender (WARMUP_N samples) — see WARMUP_N.
    for name, factory, _prep in contenders:
        Xn, yn, _ = native[name]
        factory(arch, in_shape).fit(Xn[:WARMUP_N], yn[:WARMUP_N])

    # FIT: outer loop repeats, inner loop contenders -> true round-robin.
    # A fresh estimator per repeat (fresh weights); construction is untimed.
    fit_samples = {name: [] for name, *_ in contenders}
    fitted = {}
    for _ in range(REPEATS):
        for name, factory, _prep in contenders:
            Xn, yn, _ = native[name]
            est = factory(arch, in_shape)
            t0 = time.perf_counter()
            est.fit(Xn, yn)
            fit_samples[name].append(time.perf_counter() - t0)
            fitted[name] = est

    # PREDICT: batch predict over the test subset, round-robin.
    predict_samples = {name: [] for name, *_ in contenders}
    for _ in range(PREDICT_CALLS):
        for name, *_rest in contenders:
            Xtn = native[name][2]
            t0 = time.perf_counter()
            fitted[name].predict(Xtn)
            predict_samples[name].append((time.perf_counter() - t0) * 1000.0)

    fit_s = {n: {"median": median(s), "samples": s}
             for n, s in fit_samples.items()}
    predict_ms = {n: {"median": median(s), "samples": s}
                  for n, s in predict_samples.items()}
    return fit_s, predict_ms


# --- environment -------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _env_block() -> dict:
    """Versions and thread settings — thread knobs are left at each
    framework's default and RECORDED, not equalized."""
    from mantissa_cnn import MANTISSA_MIN_VERSION
    env = {
        "cpu": _cpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mantissa": f">={MANTISSA_MIN_VERSION} (f32 CNN primitives)",
        "mantissa_threads": os.environ.get("MANTISSA_THREADS",
                                           f"default({os.cpu_count()})"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    import sklearn
    env["sklearn"] = sklearn.__version__
    import torch
    env["torch"] = torch.__version__
    env["torch_threads"] = torch.get_num_threads()
    import tensorflow as tf
    import keras
    env["tensorflow"] = tf.__version__
    env["keras"] = keras.__version__
    env["tf_inter_op_threads"] = tf.config.threading.get_inter_op_parallelism_threads()
    env["tf_intra_op_threads"] = tf.config.threading.get_intra_op_parallelism_threads()
    env["tf_threads_note"] = "0 = TensorFlow default (runtime-chosen)"
    return env


# --- entrypoint --------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--worker":
        return _run_worker(argv[1], argv[2], argv[3])

    contenders = _contenders()
    names = [n for n, *_ in contenders]
    print(f"contenders: {', '.join(names)}")

    fit_s, predict_ms, peak_rss_mb = {}, {}, {}
    t_start = time.perf_counter()
    for arch in ARCHITECTURES:
        for dataset in DATASETS:
            pair = f"{arch}/{dataset}"
            print(f"\n[{pair}] timing (R={REPEATS}, interleaved) ...")
            f, p = _time_pair(arch, dataset, contenders)
            fit_s[pair] = f
            predict_ms[pair] = p
            for name in names:
                print(f"  {name:12s} fit {f[name]['median']:8.3f} s   "
                      f"predict {p[name]['median']:9.2f} ms")
            print(f"[{pair}] peak RSS (fresh subprocess each) ...")
            peak_rss_mb[pair] = {}
            for name in names:
                mb = _measure_rss(name, arch, dataset)
                peak_rss_mb[pair][name] = round(mb, 4)
                print(f"  {name:12s} {mb:8.1f} MB")

    out = {
        "env": _env_block(),
        "protocol": {"architectures": list(ARCHITECTURES),
                     "datasets": list(DATASETS),
                     "n_train": N_TRAIN, "n_test": N_TEST,
                     "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                     "seed": SEED, "repeats": REPEATS,
                     "predict_calls": PREDICT_CALLS},
        "fit_s": fit_s,
        "predict_ms": predict_ms,
        "peak_rss_mb": peak_rss_mb,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)} "
          f"({time.perf_counter() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
