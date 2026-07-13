"""Sequential CNN classifier: mini-batch SGD on softmax cross-entropy.

Training loop design:
- Shuffled mini-batches (seeded ``np.random.default_rng``), plain SGD
  (Robbins & Monro, 1951; the mini-batch form of LeCun et al., 1998).
- The final layer must be ``Dense(classes)`` with the identity activation:
  it emits logits, and the loss applies softmax itself (fused
  softmax-cross-entropy, numerically stable via max-subtraction). Softmax as
  a layer activation would just be re-deriving the same gradient with more
  round-off.
- Memory: mini-batch input/label staging buffers are allocated once per fit
  (one pair for full batches, one for the epoch tail) and refilled with
  ``np.take(..., out=...)``; layer scratch is allocated once per batch shape
  (see :mod:`mantissa_cnn.layers`). Steady-state training does no per-batch
  allocation.

Backends: ``backend="mantissa"`` (default) runs every primitive in the
mantissa C engine and raises with the exact fix command when the engine is
missing or predates the CNN API; ``backend="numpy"`` is the pure-numpy
reference implementation (same call signatures, same semantics).

Data contract: X is NCHW float32 (scaled however you like; the datasets
module gives [0,1]), y is integer class ids 0..classes-1.
"""
from __future__ import annotations

import numpy as np

from . import _numpy_backend
from ._engine import cnn_engine

__all__ = ["Sequential"]


class Sequential:
    """A stack of layers trained with softmax cross-entropy + SGD.

    Parameters
    ----------
    layers : list of Layer (Conv2D / MaxPool2D / Flatten / Dense)
    seed : int
        Seeds one rng stream used for both weight init and epoch shuffling —
        two models with the same seed and backend train identically.
    backend : {"mantissa", "numpy"}
        "mantissa" (default) requires the C engine with CNN primitives and
        raises ImportError/RuntimeError with the exact fix otherwise.

    Fitted attributes
    -----------------
    history_ : dict with "loss" (per-epoch mean training loss) and "acc"
        (per-epoch training accuracy, accumulated on the fly from each
        mini-batch's pre-update forward pass).
    """

    def __init__(self, layers, seed: int = 0, backend: str = "mantissa"):
        if backend == "mantissa":
            self._backend = cnn_engine()      # raises with the exact fix
        elif backend == "numpy":
            self._backend = _numpy_backend
        else:
            raise ValueError(f"backend must be 'mantissa' or 'numpy', got {backend!r}")
        self.backend = backend
        self.layers = list(layers)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._built = False

    # -- construction ---------------------------------------------------------

    def build(self, in_shape):
        """Initialize parameters for input shape (c, h, w). Called by fit()
        automatically; call it yourself to inspect summary() before training."""
        shape = tuple(int(d) for d in in_shape)
        for layer in self.layers:
            shape = layer.build(shape, self._rng)
        if len(shape) != 1:
            raise ValueError(f"model output must be flat logits, got shape {shape} "
                             f"— end with Flatten()/Dense(classes)")
        self.n_classes_ = shape[0]
        self._built = True
        return self

    def summary(self) -> str:
        """Per-layer output shapes and parameter counts (build() first)."""
        if not self._built:
            raise RuntimeError("summary() needs parameters — call build(in_shape) or fit() first")
        rows = [(type(l).__name__, str(l.out_shape), l.param_count())
                for l in self.layers]
        total = sum(r[2] for r in rows)
        w = max(len(r[0]) for r in rows)
        lines = [f"{'layer':<{w}}  {'out shape':<16}  params",
                 "-" * (w + 26)]
        lines += [f"{name:<{w}}  {shape:<16}  {p:,}" for name, shape, p in rows]
        lines.append(f"total params: {total:,}")
        return "\n".join(lines)

    # -- training -------------------------------------------------------------

    def fit(self, X, y, epochs: int = 10, batch_size: int = 32,
            lr: float = 0.01, verbose: bool = False):
        X = self._check_X(X)
        y = np.ascontiguousarray(y, dtype=np.int32).ravel()
        n = len(X)
        if len(y) != n:
            raise ValueError(f"X has {n} samples but y has {len(y)}")
        if not self._built:
            self.build(X.shape[1:])
        if y.min() < 0 or y.max() >= self.n_classes_:
            raise ValueError(f"y must be class ids in [0, {self.n_classes_}); "
                             f"got range [{y.min()}, {y.max()}]")

        backend = self._backend
        bs = min(int(batch_size), n)
        classes = self.n_classes_
        self.history_ = {"loss": [], "acc": []}

        # Per-fit staging buffers, refilled in place every batch: one set for
        # full batches, one for the epoch tail. No per-batch allocation.
        tail = n % bs
        Xb = np.empty((bs,) + X.shape[1:], dtype=np.float32)
        yb = np.empty(bs, dtype=np.int32)
        dlog = np.empty((bs, classes), dtype=np.float32)
        Xt = np.empty((tail,) + X.shape[1:], dtype=np.float32) if tail else None
        yt = np.empty(tail, dtype=np.int32) if tail else None
        dlogt = np.empty((tail, classes), dtype=np.float32) if tail else None

        for epoch in range(int(epochs)):
            order = self._rng.permutation(n)
            loss_sum = 0.0
            correct = 0
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                nb = len(idx)
                bx, by, bd = (Xb, yb, dlog) if nb == bs else (Xt, yt, dlogt)
                np.take(X, idx, axis=0, out=bx)
                np.take(y, idx, out=by)

                out = bx
                for layer in self.layers:
                    out = layer.forward(out, backend)

                loss = backend.softmax_xent(out, by, bd, nb, classes)
                loss_sum += loss * nb
                correct += int(np.count_nonzero(out.argmax(axis=1) == by))

                grad = bd
                for i in range(len(self.layers) - 1, -1, -1):
                    grad = self.layers[i].backward(grad, backend, need_dx=i > 0)
                for layer in self.layers:      # after ALL grads: dX of layer i
                    layer.step(backend, lr)    # depends on its pre-step params

            self.history_["loss"].append(loss_sum / n)
            self.history_["acc"].append(correct / n)
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}  "
                      f"loss {self.history_['loss'][-1]:.4f}  "
                      f"acc {self.history_['acc'][-1]:.4f}")
        return self

    # -- inference --------------------------------------------------------------

    def _logits(self, X, chunk: int = 256):
        X = self._check_X(X, expect=self.layers[0].in_shape)
        out = np.empty((len(X), self.n_classes_), dtype=np.float32)
        for s in range(0, len(X), chunk):
            h = X[s:s + chunk]                 # contiguous slice view, no copy
            for layer in self.layers:
                h = layer.forward(h, self._backend)
            out[s:s + chunk] = h
        return out

    def predict_proba(self, X):
        """Softmax class probabilities, shape (n, classes)."""
        z = self._logits(X)
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        z /= z.sum(axis=1, keepdims=True)
        return z

    def predict(self, X):
        """Predicted class ids, shape (n,)."""
        return self._logits(X).argmax(axis=1)

    def score(self, X, y) -> float:
        """Mean accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    # -- internals ---------------------------------------------------------------

    def _check_X(self, X, expect=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 4:
            raise ValueError(f"X must be NCHW (n, c, h, w) float32, got ndim={X.ndim}")
        if expect is not None and tuple(X.shape[1:]) != tuple(expect):
            raise ValueError(f"X has sample shape {tuple(X.shape[1:])}, "
                             f"model was built for {tuple(expect)}")
        return X
