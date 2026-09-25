import numpy as np
import pytest

from gradient.data import load_digits
from gradient.layers import MLP
from gradient.metrics import accuracy
from gradient.optim import SGD, Adam, Momentum
from gradient.split import train_test_split


def quadratic(x, A, b):
    """Потери ½ xᵀAx − bᵀx и их градиент."""
    return 0.5 * x @ A @ x - b @ x, A @ x - b


class TestSGD:
    def test_step(self):
        p = np.array([1.0, 2.0])
        SGD(0.1).step([p], [np.array([1.0, -3.0])])
        np.testing.assert_allclose(p, [0.9, 2.3])

    def test_updates_in_place(self):
        p = np.zeros(3)
        before = p
        SGD(0.5).step([p], [np.ones(3)])
        assert p is before


class TestMomentum:
    def test_accumulates(self):
        p = np.zeros(2)
        opt = Momentum(0.1, 0.9)
        g = np.ones(2)
        opt.step([p], [g])
        first = p.copy()
        opt.step([p], [g])
        second = p - first
        np.testing.assert_allclose(first, [-0.1, -0.1])
        np.testing.assert_allclose(second, [-0.19, -0.19]), "v = 0.9·v + g"

    def test_state_shapes(self):
        opt = Momentum(0.1)
        params = [np.zeros((2, 3)), np.zeros(3)]
        opt.step(params, [np.ones((2, 3)), np.ones(3)])
        assert [v.shape for v in opt.velocity] == [(2, 3), (3,)]


class TestAdam:
    def test_first_step_is_about_lr(self):
        p = np.zeros(3)
        Adam(0.01).step([p], [np.array([5.0, -0.001, 100.0])])
        np.testing.assert_allclose(p, [-0.01, 0.01, -0.01], rtol=1e-3), "шаг не зависит от масштаба градиента"

    def test_bias_correction(self):
        p = np.zeros(1)
        opt = Adam(0.1, beta1=0.9, beta2=0.999, eps=0.0)
        g = np.array([2.0])
        opt.step([p], [g])
        np.testing.assert_allclose(p, [-0.1], rtol=1e-9)
        assert opt.t == 1

    def test_counter_and_state(self):
        opt = Adam()
        params = [np.zeros((2, 2))]
        for _ in range(3):
            opt.step(params, [np.ones((2, 2))])
        assert opt.t == 3
        assert opt.m[0].shape == (2, 2) and opt.v[0].shape == (2, 2)

    def test_converges_on_quadratic(self):
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(5, 5))
        A = Q.T @ Q + 5 * np.eye(5)
        b = rng.normal(size=5)
        exact = np.linalg.solve(A, b)
        for opt, steps in [(Adam(0.1), 400), (Momentum(0.01, 0.9), 400), (SGD(0.01), 400)]:
            x = np.zeros(5)
            for _ in range(steps):
                opt.step([x], [A @ x - b])
            assert np.linalg.norm(x - exact) < 0.05, f"{type(opt).__name__} не сошёлся"

    def test_adam_and_momentum_beat_sgd_on_ill_conditioned(self):
        # собственные числа 100, 1 и 0.01: вдоль последнего направления обычный спуск ползёт
        A = np.diag([100.0, 1.0, 0.01])
        b = np.ones(3)
        exact = b / np.diag(A)
        err = {}
        for name, opt in [("sgd", SGD(0.01)), ("momentum", Momentum(0.01, 0.9)), ("adam", Adam(0.05))]:
            x = np.zeros(3)
            for _ in range(6000):
                opt.step([x], [A @ x - b])
            err[name] = np.linalg.norm(x - exact)
        assert err["adam"] < err["sgd"] / 10, f"Adam {err['adam']:.3f} против SGD {err['sgd']:.3f}"
        assert err["momentum"] < err["sgd"] / 10


class TestFitWithOptimizer:
    def test_sgd_equals_plain_fit(self):
        X, y = load_digits()
        a = MLP([64, 16, 10], seed=0, init="he").fit(X[:500], y[:500], lr=0.2, epochs=3, batch_size=50, seed=0)
        b = MLP([64, 16, 10], seed=0, init="he").fit(X[:500], y[:500], lr=0.2, epochs=3, batch_size=50,
                                                    seed=0, optimizer=SGD(0.2))
        np.testing.assert_allclose(a, b)

    def test_adam_on_digits(self):
        X, y = load_digits()
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, seed=0)
        net = MLP([64, 64, 10], seed=0, init="he")
        history = net.fit(Xtr, ytr, epochs=20, batch_size=64, seed=0, optimizer=Adam(0.01))
        assert history[-1] < history[0]
        assert accuracy(yte, net.predict(Xte)) >= 0.975
