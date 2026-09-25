import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gradient
from gradient.store import load_model
from gradient.train import MODELS, evaluate

ROOT = Path(__file__).resolve().parents[2]
PNG = b"\x89PNG\r\n\x1a\n"


def test_version():
    assert gradient.__version__ == "1.0.0"


def test_models_include_cnn():
    assert set(MODELS) == {"knn", "softmax", "cnn"}


@pytest.fixture(scope="module")
def cnn_report():
    return evaluate("cnn", seed=0)


def test_cnn_quality(cnn_report):
    assert cnn_report["n_test"] == 3000, "цифры 28×28: 15 000 примеров, тест — 20 %"
    assert cnn_report["accuracy"] >= 0.965, f"точность {cnn_report['accuracy']}"
    assert len(cnn_report["history"]) == 20
    assert cnn_report["history"][-1] < cnn_report["history"][0]
    json.dumps(cnn_report)


def test_report_shape(cnn_report):
    cm = np.array(cnn_report["confusion"])
    assert cm.shape == (10, 10) and cm.sum() == 3000
    assert len(cnn_report["per_class"]) == 10


def test_saves_model_and_figures(tmp_path):
    model_path = tmp_path / "cnn.npz"
    report = evaluate("cnn", seed=0, epochs=1, save=model_path, figures=tmp_path / "fig")
    assert model_path.is_file()
    net, meta = load_model(model_path)
    assert meta["model"] == "cnn" and meta["gradient"] == "1.0.0"
    assert abs(meta["accuracy"] - report["accuracy"]) < 1e-12
    assert [type(x).__name__ for x in net.layers] == ["Conv2D", "ReLU", "MaxPool2D", "Flatten", "Dense"]
    for name in ("confusion-cnn.png", "curves-cnn.png"):
        f = tmp_path / "fig" / name
        assert f.is_file() and f.read_bytes()[:8] == PNG


def test_softmax_still_works():
    report = evaluate("softmax", seed=0, epochs=5)
    assert report["n_test"] == 359 and len(report["history"]) == 5


def test_cli(tmp_path):
    out = tmp_path / "report.json"
    res = subprocess.run([sys.executable, "-m", "gradient.train", "--model", "cnn", "--epochs", "1",
                          "--json", str(out), "--save", str(tmp_path / "m.npz"),
                          "--figures", str(tmp_path / "f")],
                         cwd=tmp_path, capture_output=True, text=True, timeout=600,
                         env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"] == "cnn"
    assert (tmp_path / "m.npz").is_file()
    assert (tmp_path / "f" / "curves-cnn.png").is_file()


def test_release_notes():
    notes = ROOT / "docs" / "release-1.0.md"
    assert notes.is_file(), "нет docs/release-1.0.md"
    text = notes.read_text(encoding="utf-8")
    for heading in ("## Точность", "## Сравнение моделей", "## Где ошибается", "## Ограничения", "## Что дальше"):
        assert heading in text, f"в описании релиза нет раздела «{heading[3:]}»"
    assert "0.9" in text or "97" in text, "в разделе «Точность» — числа из отчётов"
