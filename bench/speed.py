"""Speed + peak-RSS benchmark vs famous CNN implementations — SKELETON.

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
"""
from __future__ import annotations

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


def main() -> int:
    raise NotImplementedError("bench skeleton — implementation lands in the "
                              "benchmark phase; the docstring is the spec")


if __name__ == "__main__":
    raise SystemExit(main())
