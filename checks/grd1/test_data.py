import numpy as np

from gradient.data import DIGITS8, as_images, class_counts, load_digits


def test_default_path_is_repo_data():
    assert DIGITS8.name == "digits8.npz" and DIGITS8.is_file()


def test_shapes_and_types():
    X, y = load_digits()
    assert X.shape == (1797, 64)
    assert y.shape == (1797,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64


def test_pixels_scaled():
    X, _ = load_digits()
    assert X.min() == 0.0
    assert X.max() == 1.0
    raw = np.load(DIGITS8)["images"]
    np.testing.assert_allclose(X[5], raw[5].ravel() / 16)


def test_labels():
    _, y = load_digits()
    np.testing.assert_array_equal(y[:10], np.arange(10))
    assert set(np.unique(y)) == set(range(10))


def test_as_images_is_view():
    X, _ = load_digits()
    imgs = as_images(X)
    assert imgs.shape == (1797, 8, 8)
    assert np.shares_memory(imgs, X)
    np.testing.assert_array_equal(imgs[3, 2], X[3, 16:24])


def test_as_images_other_side():
    X = np.zeros((2, 784))
    assert as_images(X, 28).shape == (2, 28, 28)


def test_class_counts():
    _, y = load_digits()
    counts = class_counts(y)
    assert counts.shape == (10,)
    assert counts.sum() == 1797
    assert counts[0] == 178 and counts[9] == 180
    np.testing.assert_array_equal(class_counts(np.array([2, 2, 0]), 4), [1, 0, 2, 0])
