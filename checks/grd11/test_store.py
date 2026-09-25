import json
import zipfile

import numpy as np
import pytest

import gradient
from gradient.conv import Conv2D, Flatten, MaxPool2D
from gradient.layers import Dense, Dropout, MLP, ReLU, Sequential
from gradient.store import build, describe, load_model, save_model


@pytest.fixture
def net():
    rng = np.random.default_rng(0)
    net = MLP([6, 5, 3], seed=1, init="he", dropout=0.25)
    for p in net.params():
        p[...] = rng.normal(size=p.shape)
    return net


@pytest.fixture
def cnn():
    return Sequential([Conv2D(1, 4, 3, seed=0), ReLU(), MaxPool2D(2), Flatten(),
                       Dense(4 * 3 * 3, 5, seed=1, init="he")])


class TestDescribe:
    def test_mlp(self, net):
        spec = describe(net)
        assert [s["type"] for s in spec] == ["Dense", "ReLU", "Dropout", "Dense"]
        assert spec[0] == {"type": "Dense", "n_in": 6, "n_out": 5}
        assert spec[2]["p"] == 0.25
        json.dumps(spec)

    def test_cnn(self, cnn):
        spec = describe(cnn)
        assert [s["type"] for s in spec] == ["Conv2D", "ReLU", "MaxPool2D", "Flatten", "Dense"]
        assert spec[0] == {"type": "Conv2D", "c_in": 1, "c_out": 4, "k": 3}
        assert spec[2] == {"type": "MaxPool2D", "k": 2}

    def test_unknown_layer(self, net):
        class Swish:
            def params(self):
                return []

        net.layers.append(Swish())
        with pytest.raises(ValueError):
            describe(net)


class TestBuild:
    def test_roundtrip_shapes(self, cnn):
        copy = build(describe(cnn))
        assert [type(a).__name__ for a in copy.layers] == [type(a).__name__ for a in cnn.layers]
        for a, b in zip(copy.params(), cnn.params()):
            assert a.shape == b.shape

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            build([{"type": "Attention"}])


class TestSaveLoad:
    def test_predictions_match(self, net, tmp_path):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 6))
        expected = net.eval().forward(X)
        path = tmp_path / "model.npz"
        save_model(path, net, accuracy=0.97, dataset="digits8")
        loaded, meta = load_model(path)
        np.testing.assert_array_equal(loaded.eval().forward(X), expected)
        for a, b in zip(loaded.params(), net.params()):
            np.testing.assert_array_equal(a, b)

    def test_meta(self, net, tmp_path):
        path = tmp_path / "model.npz"
        save_model(path, net, accuracy=0.97, note="проверка")
        _, meta = load_model(path)
        assert meta["gradient"] == gradient.__version__
        assert meta["numpy"] == np.__version__
        assert meta["accuracy"] == 0.97 and meta["note"] == "проверка"
        assert [s["type"] for s in meta["spec"]] == ["Dense", "ReLU", "Dropout", "Dense"]

    def test_cnn_roundtrip(self, cnn, tmp_path):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(3, 1, 8, 8))
        expected = cnn.forward(X)
        path = tmp_path / "cnn.npz"
        save_model(path, cnn)
        loaded, _ = load_model(path)
        np.testing.assert_array_equal(loaded.forward(X), expected)

    def test_no_pickle_inside(self, net, tmp_path):
        path = tmp_path / "model.npz"
        save_model(path, net)
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                head = z.read(name)[:128]
                assert b"numpy.core.multiarray" not in head and b"_reconstruct" not in head
        # и читается с запретом pickle
        with np.load(path, allow_pickle=False) as f:
            assert "meta" in f.files

    def test_float32_is_smaller(self, tmp_path):
        rng = np.random.default_rng(3)
        big = MLP([200, 150, 10], seed=0)
        for p in big.params():
            p[...] = rng.normal(size=p.shape)
        wide = tmp_path / "f64.npz"
        narrow = tmp_path / "f32.npz"
        save_model(wide, big)
        save_model(narrow, big, dtype=np.float32)
        assert narrow.stat().st_size < 0.6 * wide.stat().st_size
        loaded, meta = load_model(narrow)
        assert meta["dtype"] == "float32"
        X = rng.normal(size=(10, 200))
        np.testing.assert_allclose(loaded.forward(X), big.forward(X), rtol=1e-5, atol=1e-5)
        assert loaded.params()[0].dtype == np.float64, "в сеть веса возвращаются в её типе"

    def test_shape_mismatch(self, net, tmp_path):
        path = tmp_path / "model.npz"
        save_model(path, net)
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        data["p0"] = np.zeros((7, 5))
        np.savez(tmp_path / "broken.npz", **data)
        with pytest.raises(ValueError):
            load_model(tmp_path / "broken.npz")
