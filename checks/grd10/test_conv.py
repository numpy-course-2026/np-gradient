import numpy as np
import pytest

from gradient.check import gradient_check
from gradient.conv import Conv2D, Flatten, MaxPool2D, windows
from gradient.data import load_digits28
from gradient.layers import Dense, ReLU, Sequential
from gradient.metrics import accuracy
from gradient.split import train_test_split


def naive_conv(X, W, b):
    """Эталон «в лоб»: по одному окну."""
    n, c, h, w = X.shape
    o, _, k, _ = W.shape
    out = np.zeros((n, o, h - k + 1, w - k + 1))
    for i in range(h - k + 1):
        for j in range(w - k + 1):
            patch = X[:, None, :, i:i + k, j:j + k]
            out[:, :, i, j] = (patch * W[None]).sum(axis=(2, 3, 4))
    return out + b[:, None, None]


class TestWindows:
    def test_shape_and_view(self):
        x = np.arange(2 * 3 * 6 * 7, dtype=np.float64).reshape(2, 3, 6, 7)
        w = windows(x, 3)
        assert w.shape == (2, 3, 4, 5, 3, 3)
        assert np.shares_memory(w, x), "окна должны быть view"

    def test_values(self):
        x = np.arange(25.0).reshape(1, 1, 5, 5)
        w = windows(x, 2)
        np.testing.assert_array_equal(w[0, 0, 0, 0], x[0, 0, :2, :2])
        np.testing.assert_array_equal(w[0, 0, 3, 3], x[0, 0, 3:5, 3:5])


class TestConv2D:
    def test_shapes(self):
        c = Conv2D(3, 5, 3, seed=0)
        assert c.W.shape == (5, 3, 3, 3) and c.b.shape == (5,)
        assert not c.b.any()
        out = c.forward(np.zeros((4, 3, 10, 12)))
        assert out.shape == (4, 5, 8, 10)

    def test_matches_naive(self):
        rng = np.random.default_rng(0)
        c = Conv2D(2, 3, 3, seed=1)
        c.b[...] = rng.normal(size=3)
        X = rng.normal(size=(4, 2, 7, 6))
        np.testing.assert_allclose(c.forward(X), naive_conv(X, c.W, c.b))

    def test_bias_per_channel(self):
        c = Conv2D(1, 2, 2, seed=2)
        c.W[...] = 0
        c.b[...] = [1.0, -1.0]
        out = c.forward(np.ones((1, 1, 4, 4)))
        np.testing.assert_allclose(out[0, 0], 1.0)
        np.testing.assert_allclose(out[0, 1], -1.0)

    def test_gradients(self):
        rng = np.random.default_rng(3)
        c = Conv2D(2, 3, 3, seed=4)
        X, dY = rng.normal(size=(3, 2, 6, 5)), rng.normal(size=(3, 3, 4, 3))

        def loss():
            return float((c.forward(X) * dY).sum())

        loss()
        dX = c.backward(dY)
        assert dX.shape == X.shape
        assert gradient_check(loss, c.W, c.dW) < 1e-6
        assert gradient_check(loss, c.b, c.db) < 1e-6
        assert gradient_check(loss, X, dX) < 1e-6, "неверный градиент по входу — проверьте развёрнутое ядро"


class TestMaxPool:
    def test_forward(self):
        x = np.array([[[[1.0, 5, 2, 0], [3, 4, 1, 1], [0, 0, 9, 2], [7, 1, 2, 3]]]])
        np.testing.assert_array_equal(MaxPool2D(2).forward(x), [[[[5, 2], [7, 9]]]])

    def test_backward_routes_to_max(self):
        p = MaxPool2D(2)
        x = np.array([[[[1.0, 5, 2, 0], [3, 4, 1, 1], [0, 0, 9, 2], [7, 1, 2, 3]]]])
        p.forward(x)
        g = p.backward(np.array([[[[10.0, 20], [30, 40]]]]))
        assert g.shape == x.shape
        np.testing.assert_array_equal(g[0, 0], [[0, 10, 20, 0], [0, 0, 0, 0], [0, 0, 40, 0], [30, 0, 0, 0]])

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            MaxPool2D(2).forward(np.zeros((1, 1, 5, 4)))

    def test_gradients(self):
        rng = np.random.default_rng(5)
        p = MaxPool2D(2)
        X, dY = rng.normal(size=(2, 3, 4, 4)), rng.normal(size=(2, 3, 2, 2))

        def loss():
            return float((p.forward(X) * dY).sum())

        loss()
        dX = p.backward(dY)
        assert gradient_check(loss, X, dX) < 1e-6


class TestFlatten:
    def test_forward_backward(self):
        f = Flatten()
        X = np.arange(24.0).reshape(2, 3, 2, 2)
        out = f.forward(X)
        assert out.shape == (2, 12)
        np.testing.assert_array_equal(out[1], X[1].ravel())
        back = f.backward(np.ones((2, 12)))
        assert back.shape == X.shape


class TestSequential:
    def test_arbitrary_layers(self):
        rng = np.random.default_rng(6)
        net = Sequential([Conv2D(1, 3, 3, seed=0), ReLU(), MaxPool2D(2), Flatten(),
                          Dense(3 * 3 * 3, 4, seed=1, init="he")])
        X, y = rng.normal(size=(5, 1, 8, 8)), rng.integers(0, 4, size=5)
        assert net.forward(X).shape == (5, 4)
        assert len(net.params()) == 4
        net.backward(y)
        for p, g in zip(net.params(), net.grads()):
            assert gradient_check(lambda: net.loss(X, y), p, g) < 1e-5

    def test_mlp_is_sequential(self):
        from gradient.layers import MLP
        assert isinstance(MLP([4, 3], seed=0), Sequential)


class TestDigits28:
    def test_loading(self):
        X, y = load_digits28()
        assert X.shape == (15000, 1, 28, 28) and X.dtype == np.float64
        assert y.shape == (15000,) and y.dtype == np.int64
        assert X.min() == 0.0 and X.max() == 1.0
        assert set(np.unique(y)) == set(range(10))

    def test_cnn_learns(self):
        X, y = load_digits28()
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, seed=0)
        net = Sequential([Conv2D(1, 8, 3, seed=0), ReLU(), MaxPool2D(2), Flatten(),
                          Dense(8 * 13 * 13, 10, seed=1, init="he")])
        history = net.fit(Xtr[:8000], ytr[:8000], lr=0.1, epochs=5, batch_size=64, seed=0)
        assert history[-1] < history[0]
        assert accuracy(yte, net.predict(Xte)) >= 0.93   # у эталона 0.946
