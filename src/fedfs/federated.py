"""Local training (optionally private / protected against forgetting) and federated averaging."""
from dataclasses import dataclass, field

import numpy as np

from .models import LOGISTIC, fisher_diag
from .privacy import DPConfig, noise_for_epsilon, epsilon as eps_of


@dataclass
class EWC:
    """Elastic weight consolidation: anchor weights + importance (Fisher) + strength."""
    wstar: np.ndarray
    fisher: np.ndarray
    lam: float


def steps_per_hospital(n, epochs, batch_size):
    B = min(batch_size, n)
    return epochs * int(np.ceil(n / B))


def local_train(w, Xb, y, *, epochs, lr, batch_size, rng, l2=1e-3, dp=None, ewc=None, replay=None, model=LOGISTIC):
    """A few epochs of mini-batch gradient descent at ONE hospital.

    dp     -> DP-SGD: Poisson-sample, clip each patient's gradient, add Gaussian noise.
    ewc    -> penalty pulling important weights toward the old solution.
    replay -> mix a few old examples (kept inside the hospital) into every batch.
    dp cannot be combined with ewc/replay here: both reuse data the accountant does not cover.
    """
    if dp is not None and (ewc is not None or replay is not None):
        raise ValueError('DP is not combined with EWC/replay in this study (see docs/theory.md).')
    w = w.copy()
    n = len(y)
    B = min(batch_size, n)
    q = B / n
    reg_mask = model.reg_mask(w)
    for _ in range(steps_per_hospital(n, epochs, batch_size)):
        if dp is not None:
            mask = rng.random(n) < q
            g = model.per_sample_grad(w, Xb[mask], y[mask]) if mask.any() else np.zeros((0, len(w)))
            if len(g):
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                g = g * np.minimum(1.0, dp.clip / np.maximum(norms, 1e-12))
            grad = (g.sum(0) + rng.normal(0, dp.noise_multiplier * dp.clip, size=w.shape)) / (q * n)
        else:
            idx = rng.choice(n, B, replace=False)
            Xs, ys = Xb[idx], y[idx]
            if replay is not None and len(replay[1]) > 0:
                ridx = rng.choice(len(replay[1]), min(B, len(replay[1])), replace=False)
                Xs = np.vstack([Xs, replay[0][ridx]])
                ys = np.concatenate([ys, replay[1][ridx]])
            grad = model.per_sample_grad(w, Xs, ys).mean(0)
        grad = grad + l2 * reg_mask * w
        if ewc is not None:
            grad = grad + ewc.lam * ewc.fisher * (w - ewc.wstar)
        w = w - lr * grad
    return w


@dataclass
class FedResult:
    w: np.ndarray
    history: list = field(default_factory=list)
    epsilons: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)


def fedavg(train, *, rounds, local_epochs, lr, batch_size, seed, w0=None,
           dp_epsilon=None, delta=1e-5, clip=1.0, ewc=None, replay=None, eval_fn=None, model=LOGISTIC):
    """Federated averaging.

    train: {hospital: (Xb, y)}  - never pooled; only weight vectors cross the 'network'.
    Each round: every hospital trains locally from the current global weights, the server
    averages the results weighted by patient count, and sends the blend back.
    With dp_epsilon, every hospital runs DP-SGD with a noise level calibrated so that its
    total spend over ALL rounds equals dp_epsilon (per-patient guarantee, add/remove-one).
    """
    rng = np.random.default_rng(seed)
    names = list(train)
    d = next(iter(train.values()))[0].shape[1]
    w = model.init(d, np.random.default_rng(seed + 12345)) if w0 is None else w0.copy()
    dps, eps_out, noise_out = {}, {}, {}
    if dp_epsilon is not None:
        for h in names:
            n = len(train[h][1])
            q = min(batch_size, n) / n
            T = rounds * steps_per_hospital(n, local_epochs, batch_size)
            sigma = noise_for_epsilon(dp_epsilon, q, T, delta)
            dps[h] = DPConfig(clip, sigma, delta)
            noise_out[h] = sigma
            eps_out[h] = eps_of(q, sigma, T, delta)
    sizes = np.array([len(train[h][1]) for h in names], float)
    hist = []
    for _ in range(rounds):
        locals_ = []
        for h in names:
            Xb, y = train[h]
            locals_.append(local_train(w, Xb, y, epochs=local_epochs, lr=lr, batch_size=batch_size, rng=rng,
                                       dp=dps.get(h), ewc=ewc, replay=None if replay is None else replay[h], model=model))
        w = np.average(np.stack(locals_), axis=0, weights=sizes)
        if eval_fn is not None:
            hist.append(eval_fn(w))
    return FedResult(w, hist, eps_out, noise_out)


def central_train(Xb, y, *, epochs, lr, batch_size, seed, model=LOGISTIC):
    """The 'forbidden ceiling': all patients pooled in one place. Same optimiser, no federation."""
    rng = np.random.default_rng(seed)
    w0 = model.init(Xb.shape[1], np.random.default_rng(seed + 12345))
    return local_train(w0, Xb, y, epochs=epochs, lr=lr, batch_size=batch_size, rng=rng, model=model)


def federated_fisher(w, train, model=LOGISTIC):
    """Each hospital computes the Fisher on its own data; the server averages (weighted by n)."""
    names = list(train)
    sizes = np.array([len(train[h][1]) for h in names], float)
    F = np.stack([fisher_diag(w, *train[h], model=model) for h in names])
    return np.average(F, axis=0, weights=sizes)
