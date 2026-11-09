"""Models with a shared interface: a logistic regression and a small neural network (MLP).

All weights live in ONE flat vector w, so federated averaging, DP-SGD and EWC work unchanged for both.
Inputs Xb already contain a bias column (last column of ones).
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def add_bias(X):
    return np.hstack([X, np.ones((len(X), 1))])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Logistic:
    name = 'logistic'

    def init(self, d, rng=None):
        return np.zeros(d)

    def logits(self, w, Xb):
        return Xb @ w

    def per_sample_grad(self, w, Xb, y):
        return (sigmoid(Xb @ w) - y)[:, None] * Xb

    def reg_mask(self, w):
        m = np.ones_like(w)
        m[-1] = 0.0  # do not shrink the bias
        return m


class MLP:
    """One hidden layer (ReLU). Per-patient gradients are exact (needed for DP-SGD)."""

    def __init__(self, hidden=16):
        self.H = hidden
        self.name = f'mlp{hidden}'

    def _dims(self, w):
        return (len(w) - self.H - 1) // self.H

    def init(self, d, rng):
        H = self.H
        W1 = rng.normal(0, np.sqrt(2.0 / d), (d, H))
        w2 = rng.normal(0, 1 / np.sqrt(H), H)
        return np.concatenate([W1.ravel(), w2, [0.0]])

    def _unpack(self, w):
        d, H = self._dims(w), self.H
        return w[:d * H].reshape(d, H), w[d * H:d * H + H], w[-1]

    def logits(self, w, Xb):
        W1, w2, b2 = self._unpack(w)
        return np.maximum(Xb @ W1, 0) @ w2 + b2

    def per_sample_grad(self, w, Xb, y):
        W1, w2, b2 = self._unpack(w)
        pre = Xb @ W1
        hid = np.maximum(pre, 0)
        r = sigmoid(hid @ w2 + b2) - y
        g_w2 = r[:, None] * hid
        dpre = (r[:, None] * w2[None, :]) * (pre > 0)
        g_W1 = Xb[:, :, None] * dpre[:, None, :]
        return np.hstack([g_W1.reshape(len(y), -1), g_w2, r[:, None]])

    def reg_mask(self, w):
        m = np.ones_like(w)
        m[-1] = 0.0
        return m


LOGISTIC = Logistic()


def get_model(name):
    if name == 'logistic':
        return LOGISTIC
    if name.startswith('mlp'):
        return MLP(int(name[3:] or 16))
    raise ValueError(name)


def proba(w, Xb, model=LOGISTIC):
    return sigmoid(model.logits(w, Xb))


def accuracy(w, Xb, y, model=LOGISTIC, offset=0.0):
    return float(((model.logits(w, Xb) + offset >= 0).astype(int) == y).mean())


def auc(w, Xb, y, model=LOGISTIC):
    return float(roc_auc_score(y, model.logits(w, Xb))) if len(np.unique(y)) == 2 else float('nan')


def log_loss_each(w, Xb, y, model=LOGISTIC):
    p = np.clip(proba(w, Xb, model), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_sample_grad(w, Xb, y, model=LOGISTIC):
    return model.per_sample_grad(w, Xb, y)


def fisher_diag(w, Xb, y, model=LOGISTIC):
    """Empirical Fisher (diagonal): how much each weight matters for the data seen."""
    return (model.per_sample_grad(w, Xb, y) ** 2).mean(0)


def lazy_score(y):
    return float(max(y.mean(), 1 - y.mean()))


def fit_offset(logits, y, iters=30, clip=6.0):
    """Local calibration: the single number b that best matches a hospital's own base rate.

    Fits sigmoid(logit + b) to the hospital's OWN practice patients (Newton steps). b is never shared.
    """
    b = 0.0
    for _ in range(iters):
        p = sigmoid(logits + b)
        g = (p - y).sum()
        h = (p * (1 - p)).sum() + 1e-6
        b = float(np.clip(b - g / h, -clip, clip))
    return b
