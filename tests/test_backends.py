"""Engine resolution errors (verbatim) and mantissa-vs-numpy parity.

The parity test is the acceptance test for the C engine: both backends run
the same seeded 2-step training on the same data and must agree to float32
tolerance. It skips (does not fail) while the engine or its CNN primitives
are absent — the engine work lands in parallel.
"""
import sys
import types

import numpy as np
import pytest

import mantissa_cnn._engine as eng
from mantissa_cnn.layers import Conv2D, Dense, Flatten, MaxPool2D
from mantissa_cnn.model import Sequential

MISSING_MSG = "mantissa is not installed — run: pip install mantissa-core"
TOO_OLD_MSG = ("mantissa >= 0.2.1 required for CNN primitives — "
               "run: pip install --upgrade mantissa-core")


def _engine_ready() -> bool:
    try:
        eng.cnn_engine()
        return True
    except Exception:
        return False


def test_missing_mantissa_message_verbatim(monkeypatch, tmp_path):
    monkeypatch.setattr(eng, "_tk", None)
    monkeypatch.setattr(eng, "_DEV_PYTHON_DIR", tmp_path / "nowhere")
    monkeypatch.setitem(sys.modules, "mantissa", None)   # import -> ImportError
    with pytest.raises(ImportError) as exc:
        Sequential([Flatten(), Dense(2)])                # default backend
    lines = str(exc.value).splitlines()
    assert lines[0] == MISSING_MSG
    assert "dev fallback also checked" in lines[1]


def test_too_old_mantissa_message_verbatim(monkeypatch):
    fake = types.ModuleType("mantissa")
    fake.Mantissa = type("Mantissa", (), {})             # no conv2d_forward
    monkeypatch.setattr(eng, "_tk", None)
    monkeypatch.setitem(sys.modules, "mantissa", fake)
    with pytest.raises(RuntimeError) as exc:
        Sequential([Flatten(), Dense(2)])
    assert str(exc.value) == TOO_OLD_MSG


def test_numpy_backend_needs_no_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(eng, "_tk", None)
    monkeypatch.setattr(eng, "_DEV_PYTHON_DIR", tmp_path / "nowhere")
    monkeypatch.setitem(sys.modules, "mantissa", None)
    X = np.random.default_rng(0).random((8, 1, 6, 6), dtype=np.float32)
    y = np.arange(8, dtype=np.int32) % 2
    net = Sequential([Conv2D(2, 3), Flatten(), Dense(2)], backend="numpy")
    net.fit(X, y, epochs=1, batch_size=4)


@pytest.mark.skipif(not _engine_ready(),
                    reason="mantissa engine with CNN primitives not available")
def test_backend_parity_two_training_steps():
    """Same seed, same data, 2 SGD steps: C engine == numpy oracle."""
    rng = np.random.default_rng(0)
    X = rng.random((16, 1, 10, 10), dtype=np.float32)
    y = (rng.random(16) < 0.5).astype(np.int32)

    def train(backend):
        net = Sequential([
            Conv2D(4, 3, pad=1, act="relu"),
            MaxPool2D(2),
            Conv2D(6, 3, act="tanh"),
            Flatten(),
            Dense(8, act="relu"),
            Dense(2),
        ], seed=1, backend=backend)
        net.fit(X, y, epochs=2, batch_size=16, lr=0.05)   # 1 batch = 1 step/epoch
        return net

    a, b = train("mantissa"), train("numpy")
    assert np.allclose(a.history_["loss"], b.history_["loss"], rtol=1e-4)
    for la, lb in zip(a.layers, b.layers):
        for attr in ("K", "b", "W"):
            if hasattr(la, attr):
                assert np.allclose(getattr(la, attr), getattr(lb, attr),
                                   rtol=1e-4, atol=1e-5), type(la).__name__
    assert np.allclose(a.predict_proba(X), b.predict_proba(X),
                       rtol=1e-4, atol=1e-5)
