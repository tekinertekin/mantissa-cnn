"""Sequential end-to-end on the numpy backend: learning, history, API."""
import numpy as np
import pytest

from mantissa_cnn.layers import Conv2D, Dense, Flatten, MaxPool2D
from mantissa_cnn.model import Sequential


def quadrant_task(n_per_class=40, size=8, seed=0):
    """Synthetic separable image task: class k = bright k-th quadrant."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    half = size // 2
    corners = [(0, 0), (0, half), (half, 0), (half, half)]
    for k, (r, c) in enumerate(corners):
        img = rng.normal(0.0, 0.1, size=(n_per_class, 1, size, size))
        img[:, 0, r:r + half, c:c + half] += 1.0
        X.append(img)
        y.append(np.full(n_per_class, k))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int32)
    order = rng.permutation(len(y))
    return np.ascontiguousarray(X[order]), y[order]


def small_net(seed=0):
    return Sequential([
        Conv2D(8, 3, pad=1, act="relu"),
        MaxPool2D(2),
        Flatten(),
        Dense(16, act="relu"),
        Dense(4),
    ], seed=seed, backend="numpy")


def test_fit_reaches_90pct_train_acc_on_separable_task():
    X, y = quadrant_task()
    net = small_net().fit(X, y, epochs=15, batch_size=16, lr=0.05)
    assert net.history_["acc"][-1] > 0.90
    assert net.score(X, y) > 0.90


def test_history_tracks_loss_and_acc_per_epoch():
    X, y = quadrant_task()
    net = small_net().fit(X, y, epochs=5, batch_size=16, lr=0.05)
    assert len(net.history_["loss"]) == 5 and len(net.history_["acc"]) == 5
    assert net.history_["loss"][-1] < net.history_["loss"][0]   # it learns
    assert all(0.0 <= a <= 1.0 for a in net.history_["acc"])


def test_same_seed_same_run():
    X, y = quadrant_task()
    a = small_net(seed=3).fit(X, y, epochs=2, batch_size=16, lr=0.05)
    b = small_net(seed=3).fit(X, y, epochs=2, batch_size=16, lr=0.05)
    assert a.history_["loss"] == b.history_["loss"]
    assert np.array_equal(a.layers[0].K, b.layers[0].K)


def test_predict_proba_and_predict():
    X, y = quadrant_task()
    net = small_net().fit(X, y, epochs=8, batch_size=16, lr=0.05)
    p = net.predict_proba(X[:7])
    assert p.shape == (7, 4)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(net.predict(X[:7]), p.argmax(axis=1))


def test_partial_tail_batch_is_handled():
    X, y = quadrant_task(n_per_class=10)          # 40 samples, bs 16 -> tail 8
    net = small_net().fit(X, y, epochs=2, batch_size=16, lr=0.05)
    assert len(net.history_["loss"]) == 2


def test_input_validation():
    net = small_net()
    X, y = quadrant_task(n_per_class=4)
    with pytest.raises(ValueError, match="NCHW"):
        net.fit(X.reshape(16, -1), y, epochs=1)
    with pytest.raises(ValueError, match="class ids"):
        net.fit(X, y + 7, epochs=1)
    net.fit(X, y, epochs=1)
    with pytest.raises(ValueError, match="sample shape"):
        net.predict(np.zeros((2, 1, 9, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="backend"):
        Sequential([Flatten(), Dense(2)], backend="torch")
