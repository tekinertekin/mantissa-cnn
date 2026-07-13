"""Pure-numpy reference backend, call-compatible with the mantissa CNN API.

Every function here has the exact signature of the corresponding
``mantissa.Mantissa`` method from the engine contract
(``agent/notes/cnn-engine-contract.md``), so ``Sequential`` can hold either
this module or a ``Mantissa`` instance and call it identically. All outputs
are written into caller-allocated buffers, mirroring the C convention — the
backend never allocates result arrays (scratch for im2col is the one
exception, noted below).

Convolution is im2col + GEMM (Chellapilla, Puri & Simard, 2006, "High
Performance Convolutional Neural Networks for Document Processing").
Backward passes follow LeCun et al. (1998), backprop through shared weights.
Serves as the correctness oracle for the C engine in tests, and as an
explicit ``backend="numpy"`` fallback. Dtype-agnostic on purpose: production
buffers are float32, gradient-check tests pass float64 for tight tolerances.
"""
from __future__ import annotations

import numpy as np

# Activation ids — must match mantissa's include/activations.h.
IDENTITY, STEP, SIGN, RELU, SIGMOID, TANH, GELU = range(7)


def _act(Z, act, out):
    if act == IDENTITY:
        if out is not Z:
            out[...] = Z
    elif act == RELU:
        np.maximum(Z, 0, out=out)
    elif act == TANH:
        np.tanh(Z, out=out)
    elif act == SIGMOID:
        out[...] = 1.0 / (1.0 + np.exp(-Z))
    else:
        raise ValueError(f"unsupported activation id {act}")
    return out


def _act_grad(Z, dY, act):
    """dZ = dY * act'(Z). Allocates (backward scratch, shapes are per-batch)."""
    if act == IDENTITY:
        return dY.copy()
    if act == RELU:
        return dY * (Z > 0)
    if act == TANH:
        t = np.tanh(Z)
        return dY * (1.0 - t * t)
    if act == SIGMOID:
        s = 1.0 / (1.0 + np.exp(-Z))
        return dY * s * (1.0 - s)
    raise ValueError(f"unsupported activation id {act}")


def conv2d_out_hw(in_h, in_w, kh, kw, stride, pad):
    """Output spatial size, floor semantics — same helper the engine exports."""
    return ((in_h + 2 * pad - kh) // stride + 1,
            (in_w + 2 * pad - kw) // stride + 1)


def _im2col(X, kh, kw, stride, pad, oh, ow):
    """(n, c, h, w) -> (n, c*kh*kw, oh*ow) patch matrix. kh*kw slice loop,
    vectorized over everything else."""
    n, c = X.shape[:2]
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else X
    cols = np.empty((n, c, kh, kw, oh, ow), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = Xp[:, :, i:i + stride * oh:stride,
                                  j:j + stride * ow:stride]
    return cols.reshape(n, c * kh * kw, oh * ow)


def conv2d_forward(X, K, bias, Z, Y, n, in_c, in_h, in_w,
                   out_c, kh, kw, stride, pad, act):
    oh, ow = conv2d_out_hw(in_h, in_w, kh, kw, stride, pad)
    Xv = X.reshape(n, in_c, in_h, in_w)
    Kv = K.reshape(out_c, in_c * kh * kw)
    cols = _im2col(Xv, kh, kw, stride, pad, oh, ow)
    Zv = (Z if Z is not None else Y).reshape(n, out_c, oh * ow)
    np.matmul(Kv, cols, out=Zv)
    if bias is not None:
        Zv += bias.reshape(1, out_c, 1)
    _act(Zv, act, Y.reshape(n, out_c, oh * ow))
    return Y


def conv2d_backward(X, K, Z, dY, dK, db, dX, n, in_c, in_h, in_w,
                    out_c, kh, kw, stride, pad, act):
    oh, ow = conv2d_out_hw(in_h, in_w, kh, kw, stride, pad)
    Xv = X.reshape(n, in_c, in_h, in_w)
    Kv = K.reshape(out_c, in_c * kh * kw)
    dZ = _act_grad(Z.reshape(n, out_c, oh * ow),
                   dY.reshape(n, out_c, oh * ow), act)
    cols = _im2col(Xv, kh, kw, stride, pad, oh, ow)   # recomputed, not cached
    # dK summed over the batch, per contract.
    dK.reshape(out_c, in_c * kh * kw)[...] = \
        np.matmul(dZ, cols.transpose(0, 2, 1)).sum(axis=0)
    if db is not None:
        db[...] = dZ.sum(axis=(0, 2))
    if dX is not None:
        dcols = np.matmul(Kv.T, dZ).reshape(n, in_c, kh, kw, oh, ow)
        h_p, w_p = in_h + 2 * pad, in_w + 2 * pad
        dXp = np.zeros((n, in_c, h_p, w_p), dtype=dX.dtype)
        for i in range(kh):
            for j in range(kw):
                dXp[:, :, i:i + stride * oh:stride,
                    j:j + stride * ow:stride] += dcols[:, :, i, j]
        dXv = dX.reshape(n, in_c, in_h, in_w)
        dXv[...] = dXp[:, :, pad:pad + in_h, pad:pad + in_w]


def maxpool2d(X, Y, argmax, n, c, in_h, in_w, pool, stride):
    oh = (in_h - pool) // stride + 1     # floor semantics: ragged edge dropped
    ow = (in_w - pool) // stride + 1
    Xv = X.reshape(n, c, in_h, in_w)
    Yv = Y.reshape(n, c, oh, ow)
    Av = argmax.reshape(n, c, oh, ow)
    rows = (np.arange(oh) * stride)[:, None]
    cols = (np.arange(ow) * stride)[None, :]
    for i in range(pool):
        for j in range(pool):
            v = Xv[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
            flat = ((rows + i) * in_w + (cols + j)).astype(argmax.dtype)
            if i == 0 and j == 0:
                Yv[...] = v
                Av[...] = flat
            else:
                m = v > Yv
                np.copyto(Yv, v, where=m)
                np.copyto(Av, np.broadcast_to(flat, Av.shape), where=m)
    return Y


def maxpool2d_backward(dY, argmax, dX, n, c, in_h, in_w, out_h, out_w):
    dXv = dX.reshape(n * c, in_h * in_w)
    dXv[...] = 0                          # callee zeroes, per contract
    dYv = dY.reshape(n * c, out_h * out_w)
    Av = argmax.reshape(n * c, out_h * out_w)
    np.add.at(dXv, (np.arange(n * c)[:, None], Av), dYv)


def linear_forward_batch(W, X, bias, Z, Y, n, out_dim, in_dim, act):
    Xv = X.reshape(n, in_dim)
    Wv = W.reshape(out_dim, in_dim)
    Zv = (Z if Z is not None else Y).reshape(n, out_dim)
    np.matmul(Xv, Wv.T, out=Zv)
    if bias is not None:
        Zv += bias.reshape(1, out_dim)
    _act(Zv, act, Y.reshape(n, out_dim))
    return Y


def linear_backward_batch(W, X, Z, dY, dW, db, dX, n, out_dim, in_dim, act):
    Xv = X.reshape(n, in_dim)
    Wv = W.reshape(out_dim, in_dim)
    dZ = _act_grad(Z.reshape(n, out_dim), dY.reshape(n, out_dim), act)
    dW.reshape(out_dim, in_dim)[...] = dZ.T @ Xv      # summed over the batch
    if db is not None:
        db[...] = dZ.sum(axis=0)
    if dX is not None:
        dX.reshape(n, in_dim)[...] = dZ @ Wv


def softmax_xent(logits, labels, dlogits, n, classes):
    """Fused softmax + cross-entropy: dlogits = (softmax - onehot)/n; returns
    the mean loss. Max-subtraction for numerical stability."""
    L = logits.reshape(n, classes)
    y = labels.reshape(n)
    d = dlogits.reshape(n, classes)
    m = L.max(axis=1, keepdims=True)
    e = np.exp(L - m)
    s = e.sum(axis=1, keepdims=True)
    d[...] = e / s
    idx = (np.arange(n), y)
    # -log softmax[label] == log(sum) - (logit - max)
    loss = float(np.mean(np.log(s[:, 0]) - (L[idx] - m[:, 0])))
    d[idx] -= 1.0
    d /= n
    return loss


def sgd_update(W, dW, n, lr):
    Wv = W.reshape(-1)
    Wv -= lr * dW.reshape(-1)[:n]
