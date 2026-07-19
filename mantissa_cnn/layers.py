"""CNN layers: Conv2D, MaxPool2D (plus the base Dense / Flatten, re-exported).

Conv2D and MaxPool2D are this package's spatial primitives. The fully-
connected ``Dense``, the ``Flatten`` shape adapter and the ``Layer`` contract
now live in :mod:`mantissa_nn.layers` (the shared base) and are re-exported
here so ``mantissa_cnn.layers.Dense`` etc. keep resolving.

Layers own their parameters as float32 numpy arrays and are backend-agnostic:
``forward``/``backward`` receive a backend object (a ``mantissa.Mantissa``
instance or the :mod:`mantissa_cnn._numpy_backend` module — identical call
signatures) and hand it caller-allocated buffers.

Memory design requirement: a layer allocates its Z/Y/gradient scratch ONCE
per batch shape and reuses it across batches and epochs. Scratch is keyed by
batch size in a small dict (a training run sees at most two sizes: the full
mini-batch and the epoch tail), so steady-state training does zero per-batch
allocation in the layers. Parameter gradients (dK/db) are allocated once
at build time.

Initialization (seeded via the rng handed to ``build``):
- relu -> He normal, std = sqrt(2/fan_in) (He, Zhang, Ren & Sun, 2015,
  "Delving Deep into Rectifiers", ICCV).
- tanh/sigmoid/identity -> Glorot uniform, limit sqrt(6/(fan_in+fan_out))
  (Glorot & Bengio, 2010, "Understanding the difficulty of training deep
  feedforward neural networks", AISTATS).
Biases start at zero.
"""
from __future__ import annotations

import numpy as np

# Base layers + shared init/activation helpers live in mantissa_nn.
from mantissa_nn.layers import Dense, Flatten, Layer, _act_id, _init_weights

__all__ = ["Conv2D", "MaxPool2D", "Flatten", "Dense"]


class Conv2D(Layer):
    """2-D convolution (cross-correlation, as is conventional), NCHW.

    K: (out_c, in_c, kh, kw) float32, bias: (out_c,). Output spatial size is
    (h + 2*pad - k)//stride + 1 — floor semantics, matching the engine.
    """

    def __init__(self, out_c: int, k, stride: int = 1, pad: int = 0,
                 act: str = "relu"):
        super().__init__()
        self.out_c = int(out_c)
        self.kh, self.kw = (int(k), int(k)) if np.isscalar(k) else map(int, k)
        self.stride = int(stride)
        self.pad = int(pad)
        self.act = _act_id(act)

    def build(self, in_shape, rng):
        in_c, in_h, in_w = self.in_shape = tuple(in_shape)
        oh = (in_h + 2 * self.pad - self.kh) // self.stride + 1
        ow = (in_w + 2 * self.pad - self.kw) // self.stride + 1
        if oh <= 0 or ow <= 0:
            raise ValueError(f"Conv2D: kernel {self.kh}x{self.kw} does not fit "
                             f"input {in_shape} (stride={self.stride}, pad={self.pad})")
        fan_in = in_c * self.kh * self.kw
        fan_out = self.out_c * self.kh * self.kw
        self.K = _init_weights(rng, (self.out_c, in_c, self.kh, self.kw),
                               fan_in, fan_out, self.act)
        self.b = np.zeros(self.out_c, dtype=np.float32)
        self.dK = np.empty_like(self.K)
        self.db = np.empty_like(self.b)
        self.out_shape = (self.out_c, oh, ow)
        return self.out_shape

    def param_count(self) -> int:
        return self.K.size + self.b.size

    def _alloc(self, n):
        return {"Z": np.empty((n,) + self.out_shape, dtype=np.float32),
                "Y": np.empty((n,) + self.out_shape, dtype=np.float32),
                "dX": np.empty((n,) + self.in_shape, dtype=np.float32)}

    def forward(self, X, backend):
        n = X.shape[0]
        s = self._bufs(n)
        self._X = X
        in_c, in_h, in_w = self.in_shape
        backend.conv2d_forward(X, self.K, self.b, s["Z"], s["Y"],
                               n, in_c, in_h, in_w, self.out_c,
                               self.kh, self.kw, self.stride, self.pad, self.act)
        return s["Y"]

    def backward(self, dY, backend, need_dx: bool = True):
        n = dY.shape[0]
        s = self._bufs(n)
        in_c, in_h, in_w = self.in_shape
        dX = s["dX"] if need_dx else None
        backend.conv2d_backward(self._X, self.K, s["Z"], dY,
                                self.dK, self.db, dX,
                                n, in_c, in_h, in_w, self.out_c,
                                self.kh, self.kw, self.stride, self.pad, self.act)
        return dX

    def step(self, backend, lr):
        backend.sgd_update(self.K, self.dK, self.K.size, lr)
        backend.sgd_update(self.b, self.db, self.b.size, lr)


class MaxPool2D(Layer):
    """Max pooling, floor semantics (a ragged edge is dropped, as in the
    engine contract). ``stride`` defaults to ``pool`` (non-overlapping)."""

    def __init__(self, pool: int = 2, stride=None):
        super().__init__()
        self.pool = int(pool)
        self.stride = int(stride) if stride is not None else self.pool

    def build(self, in_shape, rng):
        c, in_h, in_w = self.in_shape = tuple(in_shape)
        oh = (in_h - self.pool) // self.stride + 1
        ow = (in_w - self.pool) // self.stride + 1
        if oh <= 0 or ow <= 0:
            raise ValueError(f"MaxPool2D: pool {self.pool} does not fit input {in_shape}")
        self.out_shape = (c, oh, ow)
        return self.out_shape

    def _alloc(self, n):
        return {"Y": np.empty((n,) + self.out_shape, dtype=np.float32),
                "argmax": np.empty((n,) + self.out_shape, dtype=np.int32),
                "dX": np.empty((n,) + self.in_shape, dtype=np.float32)}

    def forward(self, X, backend):
        n = X.shape[0]
        s = self._bufs(n)
        c, in_h, in_w = self.in_shape
        backend.maxpool2d(X, s["Y"], s["argmax"], n, c, in_h, in_w,
                          self.pool, self.stride)
        return s["Y"]

    def backward(self, dY, backend, need_dx: bool = True):
        n = dY.shape[0]
        s = self._bufs(n)
        c, in_h, in_w = self.in_shape
        _, oh, ow = self.out_shape
        backend.maxpool2d_backward(dY, s["argmax"], s["dX"],
                                   n, c, in_h, in_w, oh, ow)
        return s["dX"]
