# mantissa-cnn

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)
[![Engine](https://img.shields.io/badge/engine-mantissa-00599C.svg)](https://github.com/tekinertekin/mantissa)

**Classic CNNs, with a C engine.**

A small convolutional-network classifier (`fit` / `predict` / `score`) whose
compute runs in [mantissa](https://github.com/tekinertekin/mantissa) — a
fast, memory-lean neural-network core in C. The Python layer is thin on
purpose: convolution (im2col + GEMM), pooling, the dense head, the fused
softmax-cross-entropy loss and the SGD update all execute in C on zero-copy
float32 buffers. A pure-numpy backend with the identical call signatures
(`backend="numpy"`) serves as the correctness oracle in the test suite and
as a fallback when the engine is absent.

Deliberately minimal: NCHW float32 images, integer class labels, softmax
cross-entropy, plain SGD. No autograd graph, no optimizer zoo, no data
augmentation. Layers allocate their forward/backward scratch once per batch
shape and reuse it across batches and epochs — steady-state training does no
per-batch allocation.

## Install

```sh
pip install mantissa-cnn   # after PyPI publication
```

Requires the engine `mantissa-nn >= 0.2.1` (the release that adds the CNN
primitives). From a checkout (works today, no PyPI needed): clone this repo
next to [mantissa](https://github.com/tekinertekin/mantissa), build the
engine (`make dist` there), then `pip install -e .` here — the package finds
the sibling checkout automatically.

## Quickstart

```sh
# datasets never download implicitly — fetch explicitly, once:
python -m mantissa_cnn.datasets download mnist     # or: download all
python -m mantissa_cnn.datasets list
```

```python
from mantissa_cnn import models, datasets

X_train, y_train, X_test, y_test = datasets.load("mnist")
net = models.lenet5()                    # C engine; backend="numpy" also works
print(net.summary())
net.fit(X_train, y_train, epochs=3, batch_size=32, lr=0.01, verbose=True)
print(net.score(X_test, y_test))
```

Or compose your own:

```python
from mantissa_cnn import Sequential, Conv2D, MaxPool2D, Flatten, Dense

net = Sequential([
    Conv2D(8, 3, pad=1, act="relu"),
    MaxPool2D(2),
    Flatten(),
    Dense(10),               # identity logits — softmax lives in the loss
], seed=0)
```

## Model zoo

Honest names: these are the classic architectures at small-image scale, not
the ImageNet originals.

| model | architecture | paper |
|-------|--------------|-------|
| `lenet5` | Conv 6@5x5 → pool → Conv 16@5x5 → pool → 120 → 84 → classes; faithful C1..F6 shapes (relu instead of the paper's tanh — flagged deviation) | LeCun, Bottou, Bengio & Haffner (1998), "Gradient-Based Learning Applied to Document Recognition", *Proc. IEEE* 86(11) |
| `minivgg` | [32, 32, pool] → [64, 64, pool] → 128 → classes, all 3x3 pad-1 — VGG-style blocks at CIFAR scale, **not** VGG-16 | Simonyan & Zisserman (2015), "Very Deep Convolutional Networks for Large-Scale Image Recognition", *ICLR* |
| `alexnet_small` | 64 → pool → 192 → pool → 384 → 256 → 256 → pool → 1024 → 512 → classes, 3x3 kernels — AlexNet-style at CIFAR scale, **not** the ImageNet net | Krizhevsky, Sutskever & Hinton (2012), "ImageNet Classification with Deep Convolutional Neural Networks", *NeurIPS* |

## Datasets

Five image-classification classics, all 10-class. **Nothing downloads
implicitly** — `data/` is gitignored and library code never touches the
network; missing files raise with the exact fix command.

| name | train/test | shape | source |
|------|------------|-------|--------|
| mnist | 60k / 10k | 1×28×28 | LeCun et al. (1998) |
| fashion_mnist | 60k / 10k | 1×28×28 | Xiao, Rasul & Vollgraf (2017) |
| kmnist | 60k / 10k | 1×28×28 | Clanuwat et al. (2018) |
| qmnist | 60k / 60k | 1×28×28 | Yadav & Bottou (2019) |
| cifar10 | 50k / 10k | 3×32×32 | Krizhevsky (2009) |

`datasets.load(name)` → `(X_train, y_train, X_test, y_test)`, NCHW float32
in [0, 1], int32 labels. `datasets.subset(name, n_train, n_test, seed)`
gives seeded stratified subsets (the benchmark protocol uses 2000/1000).

## Results

<!-- BEGIN:BENCH (bench/speed.py + bench/accuracy.py + bench/plots.py output; do not edit outside these markers) -->
Benchmarks coming — the harness protocol is fixed in `bench/` (2
architectures × 5 datasets, stratified 2000/1000 subsets, 3 epochs, batch
32, lr 0.01, seed 0, 5 interleaved repeats; contenders: ours / torch /
tensorflow-keras, plus scikit-learn's MLPClassifier on flattened pixels as
an explicitly-labeled non-CNN baseline). No number appears here that the
scripts did not measure.
<!-- END:BENCH -->

### Methodology

Fixed protocol for every contender: identical architectures, subsets,
epochs, batch size, learning rate and seeds. Timings are medians over
interleaved repeats on one machine (library versions and CPU recorded in the
results JSON); peak RSS is measured per contender in a fresh subprocess,
import cost included, because that is what a user pays. scikit-learn cannot
express a CNN, so its MLP entry is labeled a non-CNN baseline, not a rival.
*Measure, don't assume.*

## License

MIT — © Tekin Ertekin. Engine:
[mantissa](https://github.com/tekinertekin/mantissa), same author, MIT.
