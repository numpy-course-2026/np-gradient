import numpy as np
import pytest

from gradient.check import gradient_check
from gradient.data import load_digits
from gradient.layers import MLP, Dense, ReLU
from gradient.metrics import accuracy
from gradient.split import train_test_split


class TestDense:
    def test_shapes_and_init(self):
        d = Dense(4, 3, scale=0.1, seed=0)
        assert d.W.shape == (4, 3) and d.b.shape == (3,)
        assert not d.b.any(), "смещения начинаются с нулей"
        assert d.W.any(), "веса не должны быть нулями: одинаковые нейроны не разойдутся"
        assert abs(d.W.std() - 0.1) < 0.08

    def test_seed(self):
        np.testing.assert_array_equal(Dense(5, 4, seed=3).W, Dense(5, 4, seed=3).W)
        assert not np.array_equal(Dense(5, 4, seed=3).W, Dense(5, 4, seed=4).W)

    def test_forward(self):
        d = Dense(2, 3, seed=1)
        X = np.array([[1.0, 2.0], [0.0, -1.0]])
        np.testing.assert_allclose(d.forward(X), X @ d.W + d.b)

    def test_backward_shapes_and_values(self):
        rng = np.random.default_rng(0)
        d = Dense(4, 3, seed=2)
        X, dY = rng.normal(size=(7, 4)), rng.normal(size=(7, 3))
        d.forward(X)
        dX = d.backward(dY)
        assert dX.shape == X.shape
        np.testing.assert_allclose(dX, dY @ d.W.T)
        np.testing.assert_allclose(d.dW, X.T @ dY)
        np.testing.assert_allclose(d.db, dY.sum(axis=0))

    def test_gradients_numerically(self):
        rng = np.random.default_rng(1)
        d = Dense(3, 2, seed=5)
        X, dY = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))

        def loss():
            return float((d.forward(X) * dY).sum())

        loss()
        d.backward(dY)
        assert gradient_check(loss, d.W, d.dW) < 1e-6
        assert gradient_check(loss, d.b, d.db) < 1e-6
        assert gradient_check(lambda: loss(), X, d.backward(dY)) < 1e-6

    def test_params_and_grads(self):
        d = Dense(3, 2, seed=0)
        assert [p.shape for p in d.params()] == [(3, 2), (2,)]
        assert [g.shape for g in d.grads()] == [(3, 2), (2,)]
        assert d.params()[0] is d.W, "params должен отдавать сами массивы, а не копии"


class TestReLU:
    def test_forward_backward(self):
        r = ReLU()
        X = np.array([[-1.0, 0.0, 2.0]])
        np.testing.assert_array_equal(r.forward(X), [[0, 0, 2]])
        np.testing.assert_array_equal(r.backward(np.array([[5.0, 7.0, 9.0]])), [[0, 0, 9]])

    def test_no_params(self):
        assert list(ReLU().params()) == [] and list(ReLU().grads()) == []

    def test_does_not_modify_input(self):
        r = ReLU()
        X = np.array([[-1.0, 3.0]])
        r.forward(X)
        np.testing.assert_array_equal(X, [[-1.0, 3.0]])


class TestMLP:
    @pytest.fixture
    def small(self):
        rng = np.random.default_rng(0)
        net = MLP([4, 5, 3], scale=0.5, seed=1)
        return net, rng.normal(size=(10, 4)), rng.integers(0, 3, size=10)

    def test_structure(self, small):
        net, X, _ = small
        assert [type(x).__name__ for x in net.layers] == ["Dense", "ReLU", "Dense"]
        assert net.forward(X).shape == (10, 3)
        assert len(net.params()) == 4 and len(net.grads()) == 4

    def test_predict(self, small):
        net, X, _ = small
        P = net.predict_proba(X)
        np.testing.assert_allclose(P.sum(axis=1), 1)
        np.testing.assert_array_equal(net.predict(X), P.argmax(axis=1))

    def test_backward_matches_numeric(self, small):
        net, X, y = small
        net.forward(X)
        net.backward(y)
        for p, g in zip(net.params(), net.grads()):
            assert gradient_check(lambda: net.loss(X, y), p, g) < 1e-6

    def test_step_moves_down(self, small):
        net, X, y = small
        before = net.loss(X, y)
        net.forward(X)
        net.backward(y)
        net.step(0.1)
        assert net.loss(X, y) < before

    def test_digits_beats_softmax_regression(self):
        X, y = load_digits()
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, seed=0)
        net = MLP([64, 64, 10], scale=0.1, seed=0)
        history = net.fit(Xtr, ytr, lr=0.5, epochs=60, batch_size=64, seed=0)
        assert len(history) == 60 and history[-1] < history[0]
        assert accuracy(yte, net.predict(Xte)) >= 0.975
