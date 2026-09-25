import numpy as np
import pytest

from gradient.data import load_digits
from gradient.report import compare_optimizers, confusion_figure, learning_curves
from gradient.split import train_test_split

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits()
    return train_test_split(X, y, 0.2, seed=0)


class TestFigures:
    def test_learning_curves(self, tmp_path):
        path = tmp_path / "curves.png"
        got = learning_curves(path, {"sgd": [1.0, 0.6, 0.4], "adam": [1.0, 0.4, 0.2]})
        assert str(got) == str(path)
        assert path.read_bytes()[:8] == PNG
        assert path.stat().st_size > 5000

    def test_learning_curves_single(self, tmp_path):
        path = tmp_path / "one.png"
        learning_curves(path, {"adam": [2.0, 1.0]}, title="Только Adam")
        assert path.is_file()

    def test_confusion_figure(self, tmp_path):
        rng = np.random.default_rng(0)
        cm = rng.integers(0, 40, size=(10, 10))
        path = tmp_path / "cm.png"
        got = confusion_figure(path, cm)
        assert str(got) == str(path)
        assert path.read_bytes()[:8] == PNG
        assert path.stat().st_size > 5000

    def test_confusion_figure_small(self, tmp_path):
        path = tmp_path / "cm2.png"
        confusion_figure(path, np.array([[5, 1], [0, 4]]))
        assert path.is_file()


class TestCompare:
    @pytest.fixture(scope="class")
    def result(self, digits):
        Xtr, Xte, ytr, yte = digits
        return compare_optimizers(Xtr, ytr, Xte, yte, sizes=(64, 32, 10), epochs=10, seed=0)

    def test_structure(self, result):
        assert set(result) == {"sgd", "momentum", "adam"}
        for name, item in result.items():
            assert len(item["history"]) == 10
            assert all(isinstance(v, float) for v in item["history"])
            assert 0.0 <= item["accuracy"] <= 1.0
            assert item["seconds"] > 0
            cm = np.array(item["confusion"])
            assert cm.shape == (10, 10) and cm.sum() == 359

    def test_learning_happened(self, result):
        for name, item in result.items():
            assert item["history"][-1] < item["history"][0], f"{name} не учился"
            assert item["accuracy"] > 0.9, f"{name}: точность {item['accuracy']}"

    def test_adam_reaches_lower_loss(self, result):
        assert result["adam"]["history"][-1] < result["sgd"]["history"][-1]

    def test_deterministic(self, digits):
        Xtr, Xte, ytr, yte = digits
        a = compare_optimizers(Xtr, ytr, Xte, yte, sizes=(64, 16, 10), epochs=3, seed=1)
        b = compare_optimizers(Xtr, ytr, Xte, yte, sizes=(64, 16, 10), epochs=3, seed=1)
        assert a["adam"]["history"] == b["adam"]["history"]
        assert a["sgd"]["accuracy"] == b["sgd"]["accuracy"]
