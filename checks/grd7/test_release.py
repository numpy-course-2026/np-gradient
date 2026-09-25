import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import gradient
from gradient.train import evaluate

ROOT = Path(__file__).resolve().parents[2]


def test_version():
    parts = gradient.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), "версия вида X.Y.Z"
    assert tuple(int(p) for p in parts) >= (0, 1, 0)


@pytest.fixture(scope="module")
def reports():
    return {m: evaluate(m, seed=0) for m in ("knn", "softmax")}


def test_report_fields(reports):
    for name, r in reports.items():
        assert r["model"] == name and r["seed"] == 0
        assert r["n_train"] + r["n_test"] == 1797 and r["n_test"] == 359
        assert isinstance(r["accuracy"], float)
        assert len(r["per_class"]) == 10 and all(isinstance(v, float) for v in r["per_class"])
        assert len(r["confusion"]) == 10 and all(len(row) == 10 for row in r["confusion"])
        assert sum(map(sum, r["confusion"])) == 359
        assert isinstance(r["seconds"], float) and r["seconds"] >= 0
        json.dumps(r)   # отчёт целиком сериализуется в JSON


def test_numbers_consistent(reports):
    for r in reports.values():
        cm = r["confusion"]
        diag = sum(cm[i][i] for i in range(10))
        assert abs(r["accuracy"] - diag / 359) < 1e-12
        for i in range(10):
            assert abs(r["per_class"][i] - cm[i][i] / sum(cm[i])) < 1e-12


def test_quality(reports):
    assert reports["knn"]["accuracy"] >= 0.975
    assert reports["softmax"]["accuracy"] >= 0.95


def test_deterministic(reports):
    again = evaluate("softmax", seed=0)
    assert again["accuracy"] == reports["softmax"]["accuracy"]
    assert again["confusion"] == reports["softmax"]["confusion"]


def test_unknown_model():
    with pytest.raises(ValueError):
        evaluate("svm")


def test_cli(tmp_path):
    out = tmp_path / "report.json"
    res = subprocess.run([sys.executable, "-m", "gradient.train", "--model", "knn", "--seed", "0",
                          "--json", str(out)], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                         env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert res.returncode == 0, res.stderr
    assert "0.98" in res.stdout, f"в выводе нет точности: {res.stdout!r}"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"] == "knn" and data["n_test"] == 359


def test_cli_rejects_unknown_model(tmp_path):
    res = subprocess.run([sys.executable, "-m", "gradient.train", "--model", "svm"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=60, env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert res.returncode != 0


def test_release_notes():
    notes = ROOT / "docs" / "release-0.1.md"
    assert notes.is_file(), "нет docs/release-0.1.md"
    text = notes.read_text(encoding="utf-8")
    for heading in ("## Точность", "## Где ошибается", "## Что дальше"):
        assert heading in text, f"в описании релиза нет раздела «{heading[3:]}»"
    assert re.search(r"0\.9\d", text), "в разделе «Точность» — числа из отчёта"
