"""mantissa-cnn: a small, honest CNN classifier on the mantissa C engine.

>>> from mantissa_cnn import models, datasets
>>> X_train, y_train, X_test, y_test = datasets.load("mnist")
>>> net = models.lenet5()                       # backend="mantissa" (C engine)
>>> net.fit(X_train, y_train, epochs=3)
>>> print(net.score(X_test, y_test))

The default backend is the mantissa C engine (>= 0.2.1, CNN primitives);
``backend="numpy"`` selects the pure-numpy reference implementation.
"""
from ._engine import MANTISSA_MIN_VERSION, MANTISSA_PIP_NAME, cnn_engine, load_mantissa
from .layers import Conv2D, Dense, Flatten, MaxPool2D
from .model import Sequential
from . import models


def __getattr__(name):
    # PEP 562 lazy import: keeps `python -m mantissa_cnn.datasets ...` (the
    # documented download CLI) free of runpy's double-import warning.
    # importlib, not `from . import` — the latter re-enters this hook while
    # the submodule is mid-import and recurses.
    if name == "datasets":
        import importlib
        return importlib.import_module(".datasets", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"
__all__ = ["Sequential", "Conv2D", "MaxPool2D", "Flatten", "Dense",
           "models", "datasets", "cnn_engine", "load_mantissa",
           "MANTISSA_PIP_NAME", "MANTISSA_MIN_VERSION"]
