import numpy as np
import pytest

from gradient.check import gradient_check
from gradient.data import load_digits
from gradient.logistic import SoftmaxRegression
from gradient.metrics import accuracy
from gradient.split import train_test_split


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.integers(0, 3, size=20)
    m = SoftmaxRegression(5, 3, l2=0.1)
    m.W[...] = rng.normal(0, 0.5, size=(5, 3))
    m.b[...] = rng.normal(0, 0.5, size=3)
    return m, X, y


def test_init():
    m = SoftmaxRegression(64, 10)
    assert m.W.shape == (64, 10) and m.b.shape == (10,)
    assert not m.W.any() and not m.b.any()


def test_shapes_and_probabilities(small):
    m, X, _ = small
    assert m.logits(X).shape == (20, 3)
    P = m.predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    np.testing.assert_array_equal(m.predict(X), P.argmax(axis=1))


def test_initial_loss_is_log_k():
    X = np.random.default_rng(1).normal(size=(30, 4))
    m = SoftmaxRegression(4, 10)
    np.testing.assert_allclose(m.loss(X, np.arange(30) % 10), np.log(10))


def test_l2_term(small):
    m, X, y = small
    m0 = SoftmaxRegression(5, 3, l2=0.0)
    m0.W[...], m0.b[...] = m.W, m.b
    np.testing.assert_allclose(m.loss(X, y) - m0.loss(X, y), 0.05 * (m.W ** 2).sum())


def test_gradients_pass_check(small):
    m, X, y = small
    dW, db = m.gradients(X, y)
    assert dW.shape == m.W.shape and db.shape == m.b.shape
    assert gradient_check(lambda: m.loss(X, y), m.W, dW) < 1e-6
    assert gradient_check(lambda: m.loss(X, y), m.b, db) < 1e-6


def test_check_catches_wrong_gradient(small):
    m, X, y = small
    dW, _ = m.gradients(X, y)
    assert gradient_check(lambda: m.loss(X, y), m.W, dW * 1.1) > 1e-2
    wrong = dW.copy()
    wrong[2, 1] += 0.01
    assert gradient_check(lambda: m.loss(X, y), m.W, wrong) > 1e-3


def test_check_restores_param(small):
    m, X, y = small
    before = m.W.copy()
    gradient_check(lambda: m.loss(X, y), m.W, m.gradients(X, y)[0])
    np.testing.assert_array_equal(m.W, before)


def test_check_returns_float(small):
    m, X, y = small
    assert isinstance(gradient_check(lambda: m.loss(X, y), m.b, m.gradients(X, y)[1]), float)


def test_digits():
    X, y = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, seed=0)
    m = SoftmaxRegression(64, 10)
    history = m.fit(Xtr, ytr, lr=0.5, epochs=50, batch_size=64, seed=0)
    assert len(history) == 50
    assert all(isinstance(h, float) for h in history)
    assert history[-1] < history[0] < np.log(10)
    assert accuracy(yte, m.predict(Xte)) >= 0.95
