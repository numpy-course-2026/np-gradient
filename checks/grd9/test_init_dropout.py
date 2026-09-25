import numpy as np
import pytest

from gradient.check import gradient_check
from gradient.data import load_digits
from gradient.init import init_weights
from gradient.layers import MLP, Dense, Dropout
from gradient.metrics import accuracy
from gradient.split import train_test_split


class TestInit:
    @pytest.mark.parametrize(("scheme", "std"), [
        ("normal", 0.1), ("xavier", np.sqrt(1 / 400)), ("he", np.sqrt(2 / 400)),
    ])
    def test_scale(self, scheme, std):
        W = init_weights(400, 300, scheme, scale=0.1, rng=0)
        assert W.shape == (400, 300)
        assert abs(W.std() - std) < 0.05 * std, f"{scheme}: стандартное отклонение {W.std():.4f}, ждали {std:.4f}"
        assert abs(W.mean()) < 0.01 * std + 1e-4

    def test_depends_on_n_in_not_n_out(self):
        wide = init_weights(100, 1000, "he", rng=1).std()
        narrow = init_weights(100, 10, "he", rng=2).std()
        assert abs(wide - narrow) < 0.1 * wide

    def test_seed_and_generator(self):
        np.testing.assert_array_equal(init_weights(5, 4, "he", rng=3), init_weights(5, 4, "he", rng=3))
        assert not np.array_equal(init_weights(5, 4, "he", rng=3), init_weights(5, 4, "he", rng=4))
        rng = np.random.default_rng(0)
        assert init_weights(5, 4, "he", rng=rng).shape == (5, 4)

    def test_unknown_scheme(self):
        with pytest.raises(ValueError):
            init_weights(4, 4, "kaiming-uniform")

    def test_dense_uses_scheme(self):
        d = Dense(200, 50, seed=0, init="he")
        assert abs(d.W.std() - np.sqrt(2 / 200)) < 0.1 * np.sqrt(2 / 200)
        assert abs(Dense(200, 50, seed=0).W.std() - 0.1) < 0.01, "по умолчанию — прежняя схема normal"


class TestDropout:
    def test_zeroes_about_p(self):
        d = Dropout(0.3, seed=0)
        out = d.forward(np.ones((200, 200)))
        zeros = (out == 0).mean()
        assert abs(zeros - 0.3) < 0.02, f"занулено {zeros:.3f}, ждали около 0.3"

    def test_inverted_scaling_keeps_mean(self):
        d = Dropout(0.4, seed=1)
        X = np.full((300, 300), 2.0)
        out = d.forward(X)
        assert abs(out.mean() - 2.0) < 0.02, "оставшиеся входы делят на (1 - p), чтобы среднее не менялось"
        assert abs(out[out > 0].mean() - 2 / 0.6) < 1e-9

    def test_eval_is_identity(self):
        d = Dropout(0.5, seed=2)
        d.training = False
        X = np.arange(12.0).reshape(3, 4)
        np.testing.assert_array_equal(d.forward(X), X)
        np.testing.assert_array_equal(d.backward(np.ones((3, 4))), np.ones((3, 4)))

    def test_backward_uses_same_mask(self):
        d = Dropout(0.5, seed=3)
        X = np.ones((50, 50))
        out = d.forward(X)
        back = d.backward(np.ones((50, 50)))
        np.testing.assert_array_equal(back == 0, out == 0)
        np.testing.assert_allclose(back[back > 0], 2.0)

    def test_masks_differ_between_calls(self):
        d = Dropout(0.5, seed=4)
        a = d.forward(np.ones((40, 40)))
        b = d.forward(np.ones((40, 40)))
        assert not np.array_equal(a, b), "каждый батч — своя маска"

    def test_seeded(self):
        a = Dropout(0.5, seed=7).forward(np.ones((20, 20)))
        b = Dropout(0.5, seed=7).forward(np.ones((20, 20)))
        np.testing.assert_array_equal(a, b)

    def test_p_zero_and_bad_p(self):
        np.testing.assert_array_equal(Dropout(0.0, seed=0).forward(np.ones((5, 5))), np.ones((5, 5)))
        with pytest.raises(ValueError):
            Dropout(1.0)
        with pytest.raises(ValueError):
            Dropout(-0.1)

    def test_no_params(self):
        assert list(Dropout(0.5).params()) == [] and list(Dropout(0.5).grads()) == []


class TestMLPWithDropout:
    @pytest.fixture
    def net(self):
        return MLP([4, 6, 3], seed=1, init="he", dropout=0.3)

    def test_layers(self, net):
        assert [type(x).__name__ for x in net.layers] == ["Dense", "ReLU", "Dropout", "Dense"]

    def test_train_eval_switch(self, net):
        X = np.ones((100, 4))
        net.eval()
        np.testing.assert_array_equal(net.forward(X), net.forward(X))
        net.train()
        assert not np.array_equal(net.forward(X), net.forward(X)), "в обучении маски разные"

    def test_predict_does_not_use_dropout(self, net):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        net.train()
        first = net.predict(X)
        np.testing.assert_array_equal(first, net.predict(X))
        net.eval()
        np.testing.assert_array_equal(first, net.forward(X).argmax(axis=1))

    def test_gradients_in_eval_mode(self, net):
        rng = np.random.default_rng(2)
        X, y = rng.normal(size=(12, 4)), rng.integers(0, 3, size=12)
        net.eval()
        net.forward(X)
        net.backward(y)
        for p, g in zip(net.params(), net.grads()):
            assert gradient_check(lambda: net.loss(X, y), p, g) < 1e-6

    def test_digits(self):
        X, y = load_digits()
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, seed=0)
        net = MLP([64, 64, 10], seed=0, init="he", dropout=0.2)
        history = net.fit(Xtr, ytr, lr=0.5, epochs=120, batch_size=64, seed=0)
        assert history[-1] < history[0]
        assert accuracy(yte, net.predict(Xte)) >= 0.98
