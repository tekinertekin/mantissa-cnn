"""Shape propagation through every layer, zoo architectures, scratch reuse.
All on the numpy backend — no engine needed."""
import numpy as np
import pytest

from mantissa_cnn import _numpy_backend as B
from mantissa_cnn.layers import Conv2D, Dense, Flatten, MaxPool2D
from mantissa_cnn.model import Sequential
from mantissa_cnn.models import alexnet_small, lenet5, minivgg


def _built(layer, in_shape, seed=0):
    layer.build(in_shape, np.random.default_rng(seed))
    return layer


@pytest.mark.parametrize("in_shape, out_c, k, stride, pad, expected", [
    ((1, 28, 28), 6, 5, 1, 0, (6, 24, 24)),
    ((1, 28, 28), 6, 5, 1, 2, (6, 28, 28)),      # pad=2 keeps 28x28
    ((3, 32, 32), 32, 3, 1, 1, (32, 32, 32)),
    ((3, 32, 32), 8, 3, 2, 1, (8, 16, 16)),
    ((2, 9, 7), 4, (3, 5), 2, 0, (4, 4, 2)),     # non-square input and kernel
])
def test_conv2d_shapes(in_shape, out_c, k, stride, pad, expected):
    conv = _built(Conv2D(out_c, k, stride=stride, pad=pad), in_shape)
    assert conv.out_shape == expected
    X = np.random.default_rng(1).normal(size=(3,) + in_shape).astype(np.float32)
    Y = conv.forward(X, B)
    assert Y.shape == (3,) + expected
    assert Y.dtype == np.float32
    dX = conv.backward(Y, B)
    assert dX.shape == X.shape


@pytest.mark.parametrize("in_shape, pool, stride, expected", [
    ((6, 28, 28), 2, None, (6, 14, 14)),
    ((16, 10, 10), 2, None, (16, 5, 5)),
    ((4, 5, 5), 2, 2, (4, 2, 2)),                # ragged edge dropped (floor)
    ((4, 7, 5), 3, 2, (4, 3, 2)),
])
def test_maxpool_shapes(in_shape, pool, stride, expected):
    pl = _built(MaxPool2D(pool, stride), in_shape)
    assert pl.out_shape == expected
    X = np.random.default_rng(2).normal(size=(2,) + in_shape).astype(np.float32)
    Y = pl.forward(X, B)
    assert Y.shape == (2,) + expected
    assert pl.backward(Y, B).shape == X.shape


def test_maxpool_takes_the_max():
    pl = _built(MaxPool2D(2), (1, 4, 4))
    X = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    Y = pl.forward(X, B)
    assert np.array_equal(Y[0, 0], [[5, 7], [13, 15]])


def test_flatten_roundtrip():
    fl = _built(Flatten(), (16, 5, 5))
    assert fl.out_shape == (400,)
    X = np.random.default_rng(3).normal(size=(2, 16, 5, 5)).astype(np.float32)
    Y = fl.forward(X, B)
    assert Y.shape == (2, 400)
    assert np.array_equal(fl.backward(Y, B), X)


def test_dense_shapes_and_flat_input_required():
    d = _built(Dense(120, act="relu"), (400,))
    assert d.W.shape == (120, 400) and d.out_shape == (120,)
    X = np.random.default_rng(4).normal(size=(5, 400)).astype(np.float32)
    assert d.forward(X, B).shape == (5, 120)
    with pytest.raises(ValueError, match="Flatten"):
        Dense(10).build((16, 5, 5), np.random.default_rng(0))


def test_scratch_reused_across_batches():
    # Design requirement: Z/Y/grad buffers allocated once per batch shape.
    conv = _built(Conv2D(4, 3, pad=1), (1, 8, 8))
    X = np.random.default_rng(5).normal(size=(6, 1, 8, 8)).astype(np.float32)
    assert conv.forward(X, B) is conv.forward(X, B)          # same Y buffer
    d1 = conv.backward(conv._scratch[6]["Y"], B)
    d2 = conv.backward(conv._scratch[6]["Y"], B)
    assert d1 is d2                                          # same dX buffer


def test_lenet5_faithful_shapes_and_params():
    net = lenet5(backend="numpy")
    shapes = [l.out_shape for l in net.layers]
    assert shapes == [(6, 28, 28), (6, 14, 14), (16, 10, 10), (16, 5, 5),
                      (400,), (120,), (84,), (10,)]
    # 156 + 2416 + 48120 + 10164 + 850 — the classic LeNet-5 count.
    assert sum(l.param_count() for l in net.layers) == 61706
    assert "61,706" in net.summary()


def test_minivgg_shapes():
    net = minivgg(backend="numpy")
    assert [l.out_shape for l in net.layers] == [
        (32, 32, 32), (32, 32, 32), (32, 16, 16),
        (64, 16, 16), (64, 16, 16), (64, 8, 8),
        (4096,), (128,), (10,)]


def test_alexnet_small_shapes():
    net = alexnet_small(backend="numpy")
    assert net.layers[-1].out_shape == (10,)
    assert [l.out_shape for l in net.layers][:2] == [(64, 32, 32), (64, 16, 16)]
    assert net.layers[8].out_shape == (256 * 4 * 4,)         # flatten


@pytest.mark.parametrize("factory, in_shape", [
    (lenet5, (1, 28, 28)), (minivgg, (3, 32, 32)), (alexnet_small, (3, 32, 32))])
def test_zoo_forward_pass(factory, in_shape):
    net = factory(backend="numpy")
    X = np.random.default_rng(6).random((2,) + in_shape, dtype=np.float32)
    p = net.predict_proba(X)
    assert p.shape == (2, 10)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert net.predict(X).shape == (2,)


def test_summary_requires_build():
    net = Sequential([Flatten(), Dense(2)], backend="numpy")
    with pytest.raises(RuntimeError, match="build"):
        net.summary()
