import numpy as np
import pytest

from gradient.functional import cross_entropy, log_softmax, logsumexp, sigmoid, softmax

pytestmark = pytest.mark.filterwarnings("error")


def naive_lse(z, axis=-1):
    return np.log(np.exp(z).sum(axis=axis))


class TestLogsumexp:
    def test_matches_naive_on_small_values(self):
        z = np.random.default_rng(0).normal(size=(4, 6))
        np.testing.assert_allclose(logsumexp(z), naive_lse(z))
        np.testing.assert_allclose(logsumexp(z, axis=0), naive_lse(z, axis=0))

    def test_keepdims(self):
        z = np.ones((3, 5))
        assert logsumexp(z).shape == (3,)
        assert logsumexp(z, keepdims=True).shape == (3, 1)
        assert logsumexp(z, axis=0, keepdims=True).shape == (1, 5)
        np.testing.assert_allclose(logsumexp(z), 1 + np.log(5))

    def test_huge_and_tiny(self):
        np.testing.assert_allclose(logsumexp(np.array([1000.0, 1000.0])), 1000 + np.log(2))
        np.testing.assert_allclose(logsumexp(np.array([-1000.0, -1000.0])), -1000 + np.log(2))
        np.testing.assert_allclose(logsumexp(np.array([[800.0, -800.0]])), [800.0])

    def test_minus_infinity(self):
        # -inf — законный логит «класс невозможен»
        np.testing.assert_allclose(logsumexp(np.array([0.0, -np.inf])), 0.0)


class TestSoftmax:
    def test_rows_sum_to_one(self):
        z = np.random.default_rng(1).normal(0, 50, size=(10, 7))
        p = softmax(z)
        assert p.shape == z.shape
        np.testing.assert_allclose(p.sum(axis=1), 1)
        assert (p >= 0).all()

    def test_shift_invariance(self):
        z = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(softmax(z), softmax(z + 1000))
        np.testing.assert_allclose(softmax(z), softmax(z - 1000))

    def test_extreme(self):
        np.testing.assert_allclose(softmax(np.array([[1000.0, 0.0]])), [[1.0, 0.0]])

    def test_axis0(self):
        z = np.array([[0.0, 0.0], [np.log(3), 0.0]])
        np.testing.assert_allclose(softmax(z, axis=0), [[0.25, 0.5], [0.75, 0.5]])

    def test_log_softmax(self):
        z = np.array([[1000.0, 0.0, -1000.0]])
        ls = log_softmax(z)
        np.testing.assert_allclose(ls, [[0.0, -1000.0, -2000.0]])
        w = np.random.default_rng(2).normal(size=(3, 4))
        np.testing.assert_allclose(np.exp(log_softmax(w)), softmax(w))


class TestSigmoid:
    def test_values(self):
        np.testing.assert_allclose(sigmoid(np.array([0.0, np.log(3), -np.log(3)])), [0.5, 0.75, 0.25])

    def test_extremes(self):
        s = sigmoid(np.array([-1000.0, 1000.0, -40.0]))
        np.testing.assert_allclose(s[:2], [0.0, 1.0])
        assert s[2] > 0, "sigmoid(-40) не ноль: около 4e-18"
        np.testing.assert_allclose(s[2], np.exp(-40), rtol=1e-12)

    def test_shape(self):
        assert sigmoid(np.zeros((2, 3))).shape == (2, 3)


class TestCrossEntropy:
    def test_uniform(self):
        np.testing.assert_allclose(cross_entropy(np.zeros((4, 10)), np.array([0, 3, 9, 1])), np.log(10))

    def test_value(self):
        logits = np.array([[np.log(3), 0.0], [0.0, 0.0]])
        # p(0|x1) = 3/4, p(1|x2) = 1/2
        np.testing.assert_allclose(cross_entropy(logits, np.array([0, 1])),
                                   -(np.log(0.75) + np.log(0.5)) / 2)

    def test_confident_and_wrong_is_finite(self):
        ce = cross_entropy(np.array([[1000.0, -1000.0]]), np.array([1]))
        assert isinstance(ce, float)
        np.testing.assert_allclose(ce, 2000.0)
