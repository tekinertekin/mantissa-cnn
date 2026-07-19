"""Pure-numpy reference backend — re-exported from mantissa_nn.

The implementation now lives in the base package (:mod:`mantissa_nn._numpy_backend`);
it is re-exported here unchanged so that ``mantissa_cnn._numpy_backend`` and the
activation-id constants keep resolving for existing users and tests. See the
base module for the design notes (im2col + GEMM convolution, the caller-
allocated-buffer convention, and its role as the C engine's correctness oracle).
"""
from mantissa_nn._numpy_backend import *  # noqa: F401,F403
# Private helpers are not covered by ``*``; re-export the ones the package and
# its tests may reach through this module.
from mantissa_nn._numpy_backend import (  # noqa: F401
    _act, _act_grad, _im2col,
    IDENTITY, STEP, SIGN, RELU, SIGMOID, TANH, GELU,
)
