"""Model zoo: classic CNN architectures at small-image scale, honestly named.

Each function returns a built ``Sequential`` (parameters initialized, so
``summary()`` works immediately). All use relu convolutions and an identity
logits head — the loss applies softmax (see mantissa_cnn.model).
"""
from __future__ import annotations

from .layers import Conv2D, Dense, Flatten, MaxPool2D
from .model import Sequential

__all__ = ["lenet5", "minivgg", "alexnet_small"]


def lenet5(in_shape=(1, 28, 28), classes: int = 10, seed: int = 0,
           backend: str = "mantissa") -> Sequential:
    """LeNet-5 — LeCun, Bottou, Bengio & Haffner (1998), "Gradient-Based
    Learning Applied to Document Recognition", Proc. IEEE 86(11).

    Faithful C1..F6 shapes: Conv 6@5x5 -> pool -> Conv 16@5x5 -> pool ->
    flatten -> Dense 120 -> Dense 84 -> Dense classes. The first conv pads
    by 2, reproducing the paper's 32x32 input plane from 28x28 MNIST (C1
    stays 28x28, C3 10x10, S4 5x5, so C5/F6 see the original 400 -> 120 ->
    84 widths). Deviation, flagged: the paper used scaled tanh (and RBF
    output units); relu is the modern default here (Nair & Hinton, 2010).
    """
    return Sequential([
        Conv2D(6, 5, pad=2, act="relu"),
        MaxPool2D(2),
        Conv2D(16, 5, act="relu"),
        MaxPool2D(2),
        Flatten(),
        Dense(120, act="relu"),
        Dense(84, act="relu"),
        Dense(classes),
    ], seed=seed, backend=backend).build(in_shape)


def minivgg(in_shape=(3, 32, 32), classes: int = 10, seed: int = 0,
            backend: str = "mantissa") -> Sequential:
    """VGG-style blocks at CIFAR scale — NOT VGG-16. After Simonyan &
    Zisserman (2015), "Very Deep Convolutional Networks for Large-Scale
    Image Recognition", ICLR: stacks of 3x3 pad-1 convolutions between
    2x2 pools, channel width doubling per block. Two blocks adapted to
    32x32 inputs: [32, 32, pool] -> [64, 64, pool] -> Dense 128 -> logits.
    """
    return Sequential([
        Conv2D(32, 3, pad=1, act="relu"),
        Conv2D(32, 3, pad=1, act="relu"),
        MaxPool2D(2),
        Conv2D(64, 3, pad=1, act="relu"),
        Conv2D(64, 3, pad=1, act="relu"),
        MaxPool2D(2),
        Flatten(),
        Dense(128, act="relu"),
        Dense(classes),
    ], seed=seed, backend=backend).build(in_shape)


def alexnet_small(in_shape=(3, 32, 32), classes: int = 10, seed: int = 0,
                  backend: str = "mantissa") -> Sequential:
    """AlexNet-style at CIFAR scale — NOT the ImageNet AlexNet. After
    Krizhevsky, Sutskever & Hinton (2012), "ImageNet Classification with
    Deep Convolutional Neural Networks", NeurIPS: the widely used CIFAR
    adaptation keeps the five-conv/three-dense silhouette but shrinks
    kernels and strides for 32x32 inputs (no 11x11 stride-4 stem, no LRN,
    no dropout): 64 -> pool -> 192 -> pool -> 384 -> 256 -> 256 -> pool ->
    Dense 1024 -> Dense 512 -> logits.
    """
    return Sequential([
        Conv2D(64, 3, pad=1, act="relu"),
        MaxPool2D(2),
        Conv2D(192, 3, pad=1, act="relu"),
        MaxPool2D(2),
        Conv2D(384, 3, pad=1, act="relu"),
        Conv2D(256, 3, pad=1, act="relu"),
        Conv2D(256, 3, pad=1, act="relu"),
        MaxPool2D(2),
        Flatten(),
        Dense(1024, act="relu"),
        Dense(512, act="relu"),
        Dense(classes),
    ], seed=seed, backend=backend).build(in_shape)
