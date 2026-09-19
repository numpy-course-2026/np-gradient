import numpy as np
import pytest

from gradient.linear import LinearRegression, add_bias
from gradient.metrics import mse


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    w = np.array([1.5, -2.0, 0.0, 3.0])
    return X, w, X @ w + 7.0


def test_add_bias():
    A = add_bias(np.array([[2.0, 3.0], [4.0, 5.0]]))
    np.testing.assert_array_equal(A, [[1, 2, 3], [1, 4, 5]])


@pytest.mark.parametrize("method", ["lstsq", "normal"])
def test_recovers_exact_model(data, method):
    X, w, y = data
    m = LinearRegression(method=method).fit(X, y)
    assert isinstance(m.intercept_, float)
    assert m.coef_.shape == (4,)
    np.testing.assert_allclose(m.coef_, w, atol=1e-10)
    np.testing.assert_allclose(m.intercept_, 7.0, atol=1e-10)
    np.testing.assert_allclose(m.predict(X[:5]), y[:5])


def test_fit_returns_self(data):
    X, _, y = data
    m = LinearRegression()
    assert m.fit(X, y) is m


def test_noisy(data):
    X, w, y = data
    y = y + np.random.default_rng(1).normal(0, 0.1, size=len(y))
    m = LinearRegression().fit(X, y)
    np.testing.assert_allclose(m.coef_, w, atol=0.05)
    assert mse(y, m.predict(X)) < 0.02


def test_lstsq_on_ill_conditioned_design():
    # полином 10-й степени на [0, 1]: cond(A) ~ 1e7, у AᵀA ~ 1e14
    x = np.linspace(0, 1, 60)
    X = np.vander(x, 11, increasing=True)[:, 1:]
    w = np.arange(1, 11) * 0.5
    m = LinearRegression(method="lstsq").fit(X, X @ w + 2.0)
    np.testing.assert_allclose(m.coef_, w, atol=1e-6,
                               err_msg="lstsq должен решать через lstsq/SVD, а не через AᵀA")


@pytest.mark.parametrize("method", ["lstsq", "normal"])
def test_ridge_shrinks(data, method):
    X, _, y = data
    norms = [np.linalg.norm(LinearRegression(method, l2).fit(X, y).coef_) for l2 in (0.0, 1.0, 10.0, 100.0)]
    assert all(a > b for a, b in zip(norms, norms[1:])), f"норма коэффициентов не убывает: {norms}"


@pytest.mark.parametrize("method", ["lstsq", "normal"])
def test_ridge_does_not_penalize_intercept(data, method):
    X, _, y = data
    m = LinearRegression(method, l2=1e12).fit(X, y)
    np.testing.assert_allclose(m.coef_, 0, atol=1e-6)
    np.testing.assert_allclose(m.intercept_, y.mean(), rtol=1e-6)


def test_methods_agree_with_ridge(data):
    X, _, y = data
    a = LinearRegression("lstsq", l2=3.0).fit(X, y)
    b = LinearRegression("normal", l2=3.0).fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-8)
    np.testing.assert_allclose(a.intercept_, b.intercept_, rtol=1e-8)


def test_bad_arguments():
    with pytest.raises(ValueError):
        LinearRegression(method="gd")
    with pytest.raises(ValueError):
        LinearRegression(l2=-1.0)


def test_mse():
    v = mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 3.0]))
    assert isinstance(v, float)
    np.testing.assert_allclose(v, 4 / 3)
