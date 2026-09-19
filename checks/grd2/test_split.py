import numpy as np
import pytest

from gradient.data import load_digits
from gradient.split import batches, train_test_split


@pytest.fixture(scope="module")
def data():
    X, y = load_digits()
    ids = np.arange(len(y), dtype=np.float64)[:, None]
    return np.hstack([ids, X]), y  # столбец 0 — номер строки, чтобы следить за примерами


def ids(X):
    return X[:, 0].astype(np.int64)


def test_disjoint_and_complete(data):
    X, y = data
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, seed=0)
    all_ids = np.concatenate([ids(Xtr), ids(Xte)])
    np.testing.assert_array_equal(np.sort(all_ids), np.arange(len(y)))
    np.testing.assert_array_equal(ytr, y[ids(Xtr)])
    np.testing.assert_array_equal(yte, y[ids(Xte)])


@pytest.mark.parametrize("test_size", [0.2, 0.25, 0.5, 0.1])
def test_stratified(data, test_size):
    X, y = data
    _, _, ytr, yte = train_test_split(X, y, test_size, seed=1)
    counts = np.bincount(y)
    expected = np.floor(counts * test_size + 0.5).astype(int)
    np.testing.assert_array_equal(np.bincount(yte, minlength=10), expected)
    np.testing.assert_array_equal(np.bincount(ytr, minlength=10), counts - expected)


def test_deterministic_and_seeded(data):
    X, y = data
    a = train_test_split(X, y, 0.2, seed=7)
    b = train_test_split(X, y, 0.2, seed=7)
    c = train_test_split(X, y, 0.2, seed=8)
    np.testing.assert_array_equal(ids(a[1]), ids(b[1]))
    assert set(ids(a[1])) != set(ids(c[1]))


def test_shuffled_not_sorted_by_class(data):
    X, y = data
    _, _, ytr, yte = train_test_split(X, y, 0.2, seed=0)
    assert (np.diff(ytr) != 0).sum() > 100, "обучающая выборка не перемешана: классы идут блоками"
    assert (np.diff(yte) != 0).sum() > 30


def test_batches_in_order_are_views():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10)
    got = list(batches(X, y, 4))
    assert [len(b[0]) for b in got] == [4, 4, 2]
    np.testing.assert_array_equal(got[1][1], [4, 5, 6, 7])
    assert all(np.shares_memory(bx, X) for bx, _ in got), "без перемешивания батч — срез, а не копия"


def test_batches_shuffled():
    X = np.arange(40.0).reshape(20, 2)
    y = np.arange(20)
    got = list(batches(X, y, 6, seed=3))
    assert [len(b[1]) for b in got] == [6, 6, 6, 2]
    seen = np.concatenate([b[1] for b in got])
    np.testing.assert_array_equal(np.sort(seen), np.arange(20))
    assert not (seen == np.arange(20)).all()
    for bx, by in got:
        np.testing.assert_array_equal(bx[:, 0], by * 2)
    again = np.concatenate([b[1] for b in batches(X, y, 6, seed=3)])
    np.testing.assert_array_equal(seen, again)


def test_batches_is_generator():
    X = np.zeros((10**6, 1))
    it = batches(X, np.zeros(10**6), 10, seed=0)
    bx, _ = next(it)
    assert bx.shape == (10, 1)
