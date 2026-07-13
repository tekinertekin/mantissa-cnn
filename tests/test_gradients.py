"""Numpy-backend gradients vs central finite differences.

The numpy backend is the correctness oracle for the C engine, so it must
itself be verified against the only thing more trustworthy: f(x+h) - f(x-h).
Buffers are float64 here (the backend is dtype-agnostic; production is
float32) so tolerances can be tight. Scalar loss: sum(Y * R) with fixed
random R, i.e. dY = R.
"""
import numpy as np
import pytest

from mantissa_cnn import _numpy_backend as B

EPS = 1e-6
RTOL = 1e-5
ATOL = 1e-8


def _fd(f, x, eps=EPS):
    g = np.empty_like(x)
    flat, gf = x.reshape(-1), g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        hi = f()
        flat[i] = old - eps
        lo = f()
        flat[i] = old
        gf[i] = (hi - lo) / (2 * eps)
    return g


@pytest.mark.parametrize("stride, pad", [(1, 0), (1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("act", [B.IDENTITY, B.TANH, B.RELU])
def test_conv2d_backward(stride, pad, act):
    rng = np.random.default_rng(42)
    n, in_c, in_h, in_w, out_c, kh, kw = 2, 2, 6, 5, 3, 3, 2
    oh = (in_h + 2 * pad - kh) // stride + 1
    ow = (in_w + 2 * pad - kw) // stride + 1
    X = rng.normal(size=(n, in_c, in_h, in_w))
    K = rng.normal(size=(out_c, in_c, kh, kw))
    b = rng.normal(size=out_c)
    R = rng.normal(size=(n, out_c, oh, ow))
    Z = np.empty_like(R)
    Y = np.empty_like(R)

    def loss():
        B.conv2d_forward(X, K, b, Z, Y, n, in_c, in_h, in_w,
                         out_c, kh, kw, stride, pad, act)
        return float((Y * R).sum())

    loss()
    dK, db, dX = np.empty_like(K), np.empty_like(b), np.empty_like(X)
    B.conv2d_backward(X, K, Z, R, dK, db, dX, n, in_c, in_h, in_w,
                      out_c, kh, kw, stride, pad, act)
    assert np.allclose(dK, _fd(loss, K), rtol=RTOL, atol=ATOL)
    assert np.allclose(db, _fd(loss, b), rtol=RTOL, atol=ATOL)
    assert np.allclose(dX, _fd(loss, X), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("act", [B.IDENTITY, B.TANH, B.RELU, B.SIGMOID])
def test_linear_backward_batch(act):
    rng = np.random.default_rng(7)
    n, out_dim, in_dim = 4, 3, 5
    W = rng.normal(size=(out_dim, in_dim))
    X = rng.normal(size=(n, in_dim))
    b = rng.normal(size=out_dim)
    R = rng.normal(size=(n, out_dim))
    Z, Y = np.empty_like(R), np.empty_like(R)

    def loss():
        B.linear_forward_batch(W, X, b, Z, Y, n, out_dim, in_dim, act)
        return float((Y * R).sum())

    loss()
    dW, db, dX = np.empty_like(W), np.empty_like(b), np.empty_like(X)
    B.linear_backward_batch(W, X, Z, R, dW, db, dX, n, out_dim, in_dim, act)
    assert np.allclose(dW, _fd(loss, W), rtol=RTOL, atol=ATOL)
    assert np.allclose(db, _fd(loss, b), rtol=RTOL, atol=ATOL)
    assert np.allclose(dX, _fd(loss, X), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pool, stride, in_h, in_w", [
    (2, 2, 6, 6), (2, 2, 5, 5), (3, 2, 7, 6)])   # incl. ragged edges
def test_maxpool_backward(pool, stride, in_h, in_w):
    rng = np.random.default_rng(11)
    n, c = 2, 3
    oh = (in_h - pool) // stride + 1
    ow = (in_w - pool) // stride + 1
    X = rng.normal(size=(n, c, in_h, in_w))     # continuous: ties have prob 0
    R = rng.normal(size=(n, c, oh, ow))
    Y = np.empty_like(R)
    argmax = np.empty((n, c, oh, ow), dtype=np.int32)

    def loss():
        B.maxpool2d(X, Y, argmax, n, c, in_h, in_w, pool, stride)
        return float((Y * R).sum())

    loss()
    dX = np.empty_like(X)
    B.maxpool2d_backward(R, argmax, dX, n, c, in_h, in_w, oh, ow)
    assert np.allclose(dX, _fd(loss, X), rtol=RTOL, atol=ATOL)


def test_softmax_xent_gradient_and_loss():
    rng = np.random.default_rng(13)
    n, classes = 5, 4
    logits = rng.normal(size=(n, classes)) * 3
    labels = rng.integers(0, classes, size=n).astype(np.int32)
    d = np.empty_like(logits)

    def loss():
        return B.softmax_xent(logits, labels, np.empty_like(logits), n, classes)

    ref = loss()
    # reference loss from scipy-free log-sum-exp
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    assert np.isclose(ref, np.mean(lse - logits[np.arange(n), labels]))
    B.softmax_xent(logits, labels, d, n, classes)
    assert np.allclose(d, _fd(loss, logits), rtol=RTOL, atol=ATOL)


def test_softmax_xent_stability_at_large_logits():
    logits = np.array([[1000.0, 0.0], [-1000.0, 0.0]])
    labels = np.array([0, 1], dtype=np.int32)
    d = np.empty_like(logits)
    loss = B.softmax_xent(logits, labels, d, 2, 2)
    assert np.isfinite(loss) and np.isfinite(d).all()
    assert loss < 1e-6                       # both samples are certain and right


def test_sgd_update_in_place():
    W = np.ones(6, dtype=np.float32)
    dW = np.full(6, 2.0, dtype=np.float32)
    B.sgd_update(W, dW, 6, 0.5)
    assert np.allclose(W, 0.0)
