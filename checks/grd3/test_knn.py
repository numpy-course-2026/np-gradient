import numpy as np
import pytest

from gradient.data import load_digits
from gradient.knn import knn_predict
from gradient.metrics import accuracy, confusion_matrix
from gradient.preprocess import Standardizer, one_hot
from gradient.split import train_test_split


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits()
    return train_test_split(X, y, 0.2, seed=0)


class TestStandardizer:
    def test_zero_mean_unit_std(self):
        X = np.random.default_rng(0).normal(5, 3, size=(200, 4))
        Z = Standardizer().fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-12)

    def test_fit_returns_self_and_stats(self):
        X = np.array([[1.0, 10.0], [3.0, 10.0]])
        s = Standardizer()
        assert s.fit(X) is s
        np.testing.assert_allclose(s.mean_, [2, 10])
        np.testing.assert_allclose(s.std_, [1, 0])

    @pytest.mark.filterwarnings("error")
    def test_constant_columns_become_zero(self):
        X = np.array([[0.0, 1.0, 5.0], [0.0, 3.0, 5.0], [0.0, 5.0, 5.0]])
        Z = Standardizer().fit_transform(X)
        assert np.isfinite(Z).all(), "в постоянных столбцах получились nan/inf"
        np.testing.assert_array_equal(Z[:, [0, 2]], 0)

    @pytest.mark.filterwarnings("error")
    def test_uses_train_statistics(self, digits):
        Xtr, Xte, _, _ = digits
        s = Standardizer().fit(Xtr)
        Z = s.transform(Xte)
        assert Z.shape == Xte.shape
        assert np.isfinite(Z).all()
        np.testing.assert_allclose(Z[:, 20], (Xte[:, 20] - Xtr[:, 20].mean()) / Xtr[:, 20].std())


class TestOneHot:
    def test_basic(self):
        np.testing.assert_array_equal(one_hot(np.array([2, 0, 1])), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    def test_n_classes_and_dtype(self):
        h = one_hot(np.array([1, 1]), 4)
        assert h.dtype == np.float64
        assert h.shape == (2, 4)
        np.testing.assert_array_equal(h.sum(axis=1), 1)


class TestKnn:
    def test_tiny(self):
        Xtr = np.array([[0.0], [1.0], [10.0], [11.0], [12.0]])
        ytr = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(knn_predict(Xtr, ytr, np.array([[0.4], [10.6], [6.0]]), k=3),
                                      [0, 1, 1])

    def test_vote_tie_picks_smaller_label(self):
        Xtr = np.array([[0.0], [2.0]])
        ytr = np.array([5, 3])
        np.testing.assert_array_equal(knn_predict(Xtr, ytr, np.array([[0.9]]), k=2), [3])

    def test_k1_is_nearest(self):
        rng = np.random.default_rng(2)
        Xtr = rng.normal(size=(50, 3))
        ytr = rng.integers(0, 4, size=50)
        np.testing.assert_array_equal(knn_predict(Xtr, ytr, Xtr[:10], k=1), ytr[:10])

    def test_digits_accuracy(self, digits):
        Xtr, Xte, ytr, yte = digits
        pred = knn_predict(Xtr, ytr, Xte, k=3)
        assert pred.shape == yte.shape
        assert accuracy(yte, pred) >= 0.975


class TestMetrics:
    def test_accuracy(self):
        a = accuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 4]))
        assert isinstance(a, float) and a == 0.75

    def test_confusion_matrix(self):
        cm = confusion_matrix(np.array([0, 0, 1, 2, 2, 2]), np.array([0, 1, 1, 2, 0, 2]), 3)
        np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [1, 0, 2]])
        assert cm.dtype == np.int64
