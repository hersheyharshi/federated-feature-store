"""All experiments: baselines, calibration, epsilon sweep, store privacy, drift/forgetting, membership test."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import feature_store as fs
from .data import split_hospitals
from .drift import simulate_drift, detect
from .federated import (fedavg, central_train, local_train, EWC, federated_fisher, steps_per_hospital)
from .models import (add_bias, accuracy, auc, lazy_score, log_loss_each, get_model, fit_offset)
from .privacy import DPConfig, noise_for_epsilon


@dataclass
class Config:
    seeds: tuple = tuple(range(10))
    model: str = 'logistic'
    rounds: int = 20
    local_epochs: int = 2
    lr: float = 0.1
    batch_size: int = 32
    central_epochs: int = 40
    eps_grid: tuple = (0.5, 1, 2, 4, 8)
    delta: float = 1e-5
    clip: float = 1.0
    store_train_eps: float = 4.0
    store_eps_grid: tuple = (100, 30, 10, 3, 1)
    drift_rounds_a: int = 20
    drift_rounds_b: int = 15
    drift_p_recover: float = 0.9
    drift_feature: str = None
    drift_quantile: float = 0.5
    ewc_lams: tuple = (1, 10, 100)  # lr*lambda*Fisher must stay small or training diverges
    severities: tuple = (0.3, 0.6, 0.9)


def _kw(cfg):
    return dict(local_epochs=cfg.local_epochs, lr=cfg.lr, batch_size=cfg.batch_size)


def _design(tf, X):
    return add_bias(tf.transform(X))


def _per_hospital(w, tests, model, metric=accuracy):
    return {h: metric(w, Xb, y, model=model) for h, (Xb, y) in tests.items()}


def _calibrated(w, train, tests, model):
    """Each hospital fits ONE offset on its own practice patients, then is scored with it."""
    out = {}
    for h, (Xb, y) in tests.items():
        b = fit_offset(model.logits(w, train[h][0]), train[h][1])
        out[h] = accuracy(w, Xb, y, model=model, offset=b)
    return out


def _macro(d):
    return float(np.mean(list(d.values())))


def run_baselines(frames, cfg, seed):
    model = get_model(cfg.model)
    sp = split_hospitals(frames, seed=42 + seed)
    names = list(sp)
    local_tf = {h: fs.FeatureStore.fit_local(sp[h]['X_train']) for h in names}
    shared = fs.FeatureStore.from_stats([fs.local_stats(sp[h]['X_train']) for h in names])
    kw = _kw(cfg)
    helpers = {}
    for i, h in enumerate(names):
        Xb, y = _design(local_tf[h], sp[h]['X_train']), sp[h]['y_train']
        helpers[h] = central_train(Xb, y, epochs=cfg.central_epochs, lr=cfg.lr, batch_size=cfg.batch_size,
                                   seed=seed * 100 + i, model=model)
    grid = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            grid.loc[a, b] = accuracy(helpers[a], _design(local_tf[a], sp[b]['X_test']), sp[b]['y_test'], model=model)
    local_auc = {h: auc(helpers[h], _design(local_tf[h], sp[h]['X_test']), sp[h]['y_test'], model=model) for h in names}

    Xp = np.vstack([_design(shared, sp[h]['X_train']) for h in names])
    yp = np.concatenate([sp[h]['y_train'] for h in names])
    w_pool = central_train(Xp, yp, epochs=cfg.central_epochs, lr=cfg.lr, batch_size=cfg.batch_size, seed=seed, model=model)
    tr_sh = {h: (_design(shared, sp[h]['X_train']), sp[h]['y_train']) for h in names}
    t_sh = {h: (_design(shared, sp[h]['X_test']), sp[h]['y_test']) for h in names}
    tr_loc = {h: (_design(local_tf[h], sp[h]['X_train']), sp[h]['y_train']) for h in names}
    t_loc = {h: (_design(local_tf[h], sp[h]['X_test']), sp[h]['y_test']) for h in names}
    w_nostore = fedavg(tr_loc, rounds=cfg.rounds, seed=seed, model=model, **kw).w
    w_store = fedavg(tr_sh, rounds=cfg.rounds, seed=seed, model=model, **kw).w

    acc = pd.DataFrame({
        'Local, at home': {h: grid.loc[h, h] for h in names},
        'Local, away (avg of other 3)': {h: grid.loc[h, [m for m in names if m != h]].mean() for h in names},
        'Pooled (ceiling)': _per_hospital(w_pool, t_sh, model),
        'Pooled + local calibration': _calibrated(w_pool, tr_sh, t_sh, model),
        'Federated, own scaling': _per_hospital(w_nostore, t_loc, model),
        'Federated, shared feature store': _per_hospital(w_store, t_sh, model),
        'Federated + store + local calibration': _calibrated(w_store, tr_sh, t_sh, model),
        'Lazy guess': {h: lazy_score(sp[h]['y_test']) for h in names},
    }).T
    aucs = pd.DataFrame({
        'Local, at home': local_auc,
        'Pooled (ceiling)': _per_hospital(w_pool, t_sh, model, auc),
        'Federated, own scaling': _per_hospital(w_nostore, t_loc, model, auc),
        'Federated, shared feature store': _per_hospital(w_store, t_sh, model, auc),
    }).T
    for t in (acc, aucs):
        t['Average'] = t[names].mean(axis=1)
    return dict(grid=grid, acc=acc, auc=aucs)


def run_epsilon_sweep(frames, cfg, seed, store_epsilon=None):
    model = get_model(cfg.model)
    sp = split_hospitals(frames, seed=42 + seed)
    names = list(sp)
    rng = np.random.default_rng(seed)
    shared = fs.FeatureStore.from_stats([fs.local_stats(sp[h]['X_train']) for h in names], epsilon=store_epsilon, rng=rng)
    tr = {h: (_design(shared, sp[h]['X_train']), sp[h]['y_train']) for h in names}
    te = {h: (_design(shared, sp[h]['X_test']), sp[h]['y_test']) for h in names}
    rows = []
    for eps in (None,) + tuple(cfg.eps_grid):
        res = fedavg(tr, rounds=cfg.rounds, seed=seed, dp_epsilon=eps, delta=cfg.delta, clip=cfg.clip, model=model, **_kw(cfg))
        accs, cal = _per_hospital(res.w, te, model), _calibrated(res.w, tr, te, model)
        for h in names:
            rows.append(dict(seed=seed, epsilon=np.inf if eps is None else eps, hospital=h, accuracy=accs[h],
                             accuracy_cal=cal[h], noise_multiplier=res.noise.get(h, 0.0)))
    return pd.DataFrame(rows)


def run_store_privacy(frames, cfg, seed):
    """Cost of making the shared feature-store summaries private too (training epsilon fixed)."""
    model = get_model(cfg.model)
    sp = split_hospitals(frames, seed=42 + seed)
    names = list(sp)
    rows = []
    for se in (None,) + tuple(cfg.store_eps_grid):
        rng = np.random.default_rng(seed + 7)
        shared = fs.FeatureStore.from_stats([fs.local_stats(sp[h]['X_train']) for h in names], epsilon=se, rng=rng)
        tr = {h: (_design(shared, sp[h]['X_train']), sp[h]['y_train']) for h in names}
        te = {h: (_design(shared, sp[h]['X_test']), sp[h]['y_test']) for h in names}
        res = fedavg(tr, rounds=cfg.rounds, seed=seed, dp_epsilon=cfg.store_train_eps, delta=cfg.delta,
                     clip=cfg.clip, model=model, **_kw(cfg))
        rows.append(dict(seed=seed, store_epsilon=np.inf if se is None else se,
                         total_epsilon=cfg.store_train_eps + (0 if se is None else se),
                         accuracy=_macro(_per_hospital(res.w, te, model))))
    return pd.DataFrame(rows)


def run_drift(frames, cfg, seed):
    model = get_model(cfg.model)
    rng = np.random.default_rng(seed)
    names = list(frames)
    stages = {}
    for h in names:
        df = frames[h].sample(frac=1.0, random_state=1000 + seed).reset_index(drop=True)
        half = len(df) // 2
        stages[h] = {'A': df.iloc[:half], 'B': df.iloc[half:]}
    sp = {s: split_hospitals({h: stages[h][s] for h in names}, seed=7 + seed) for s in 'AB'}
    store = fs.FeatureStore.from_stats([fs.local_stats(sp['A'][h]['X_train']) for h in names])
    Z = {s: {h: store.transform(sp[s][h]['X_train']) for h in names} for s in 'AB'}
    ZA = np.vstack([Z['A'][h] for h in names])
    w_probe = central_train(add_bias(ZA), np.concatenate([sp['A'][h]['y_train'] for h in names]),
                            epochs=cfg.central_epochs, lr=0.1, batch_size=cfg.batch_size, seed=seed)
    if cfg.drift_feature is None:
        cont = [fs.NAMES.index(n) for n in ('age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'st_depression')]
        col = cont[int(np.argmax(np.abs(w_probe[cont])))]
        sign = float(np.sign(w_probe[col]) or 1.0)
    else:
        col, sign = fs.NAMES.index(cfg.drift_feature), 1.0
    thr = np.quantile(ZA[:, col] * sign, cfg.drift_quantile)
    drift_marker = fs.NAMES[col]

    def drifted(h, which):
        Xdf = sp['B'][h][f'X_{which}']
        return simulate_drift(store.transform(Xdf)[:, col] * sign, sp['B'][h][f'y_{which}'], rng,
                              threshold=thr, p_recover=cfg.drift_p_recover)

    trainA = {h: (add_bias(Z['A'][h]), sp['A'][h]['y_train']) for h in names}
    trB, teB, deltas = {}, {}, []
    for h in names:
        yb, _ = drifted(h, 'train')
        trB[h] = (add_bias(Z['B'][h]), yb)
        yt, d2 = drifted(h, 'test')
        teB[h] = (add_bias(store.transform(sp['B'][h]['X_test'])), yt)
        deltas.append(d2)
    teA = {h: (add_bias(store.transform(sp['A'][h]['X_test'])), sp['A'][h]['y_test']) for h in names}
    kw = _kw(cfg)

    def ev(w):
        return dict(A=_macro(_per_hospital(w, teA, model)), B=_macro(_per_hospital(w, teB, model)))

    resA = fedavg(trainA, rounds=cfg.drift_rounds_a, seed=seed, eval_fn=ev, model=model, **kw)
    wA = resA.w
    before = ev(wA)
    sA = [(x, y) for h in names for x, y in zip(*teA[h])]
    sB = [(x, y) for h in names for x, y in zip(*teB[h])]
    rng.shuffle(sA)
    rng.shuffle(sB)
    errs = [int((model.logits(wA, x[None, :])[0] >= 0) != y) for x, y in sA + sB]
    alarm = detect(errs)
    det = dict(drift_at=len(sA), alarm_at=alarm,
               delay=None if alarm is None or alarm < len(sA) else alarm - len(sA),
               false_alarm=alarm is not None and alarm < len(sA))
    replay = {h: (trainA[h][0], trainA[h][1]) for h in names}
    fisher = federated_fisher(wA, trainA, model=model)
    strategies = {'No protection': {}, 'Replay': dict(replay=replay)}
    for lam in cfg.ewc_lams:
        strategies[f'EWC (lambda={lam})'] = dict(ewc=EWC(wA.copy(), fisher, lam))
    out, curves = [], {}
    for name, extra in strategies.items():
        res = fedavg(trB, rounds=cfg.drift_rounds_b, seed=seed + 1, w0=wA, eval_fn=ev, model=model, **kw, **extra)
        after = ev(res.w)
        curves[name] = res.history
        out.append(dict(seed=seed, strategy=name, acc_old_before=before['A'], acc_new_before=before['B'],
                        acc_old_after=after['A'], acc_new_after=after['B'],
                        forgetting=before['A'] - after['A'], both_stages_after=(after['A'] + after['B']) / 2))
    teB_old = {h: (teB[h][0], sp['B'][h]['y_test']) for h in names}
    gap = (1 - before['B']) - (1 - _macro(_per_hospital(wA, teB_old, model)))
    bound = dict(seed=seed, marker=drift_marker, delta=float(np.mean(deltas)),
                 error_increase_same_patients=gap, bound_holds=bool(gap <= float(np.mean(deltas)) + 1e-9))
    return dict(summary=pd.DataFrame(out), detection=det, bound=bound, curves=curves, phase_a=resA.history)


def run_membership_test(frames, cfg, seed, eps=1.0):
    """Loss-threshold membership guessing. AUC 0.5 = attacker is guessing."""
    model = get_model(cfg.model)
    sp = split_hospitals(frames, seed=42 + seed)
    lr = 0.5 if model.name == 'logistic' else 0.1
    rows = []
    for i, h in enumerate(sp):
        tf = fs.FeatureStore.fit_local(sp[h]['X_train'])
        Xtr, ytr = _design(tf, sp[h]['X_train']), sp[h]['y_train']
        Xte, yte = _design(tf, sp[h]['X_test']), sp[h]['y_test']
        rng = np.random.default_rng(seed * 10 + i)
        n, E = len(ytr), 300
        w0 = model.init(Xtr.shape[1], np.random.default_rng(seed * 10 + i + 5))
        w_plain = local_train(w0, Xtr, ytr, epochs=E, lr=lr, batch_size=cfg.batch_size, rng=rng, l2=0, model=model)
        q = min(cfg.batch_size, n) / n
        sigma = noise_for_epsilon(eps, q, steps_per_hospital(n, E, cfg.batch_size), cfg.delta)
        w_dp = local_train(w0, Xtr, ytr, epochs=E, lr=lr, batch_size=cfg.batch_size, rng=rng, l2=0, model=model,
                           dp=DPConfig(cfg.clip, sigma, cfg.delta))
        for tag, w in (('no privacy', w_plain), (f'DP eps={eps:g}', w_dp)):
            lm, ln = log_loss_each(w, Xtr, ytr, model), log_loss_each(w, Xte, yte, model)
            score = -np.concatenate([lm, ln])
            member = np.concatenate([np.ones(len(lm)), np.zeros(len(ln))])
            rows.append(dict(seed=seed, hospital=h, model=tag, attack_auc=float(roc_auc_score(member, score)),
                             train_acc=accuracy(w, Xtr, ytr, model=model), test_acc=accuracy(w, Xte, yte, model=model)))
    return pd.DataFrame(rows)


def mean_std(dfs):
    stack = np.stack([d.to_numpy(float) for d in dfs])
    m = pd.DataFrame(stack.mean(0), index=dfs[0].index, columns=dfs[0].columns)
    s = pd.DataFrame(stack.std(0, ddof=1), index=dfs[0].index, columns=dfs[0].columns)
    return m, s


def fmt(m, s, pct=True):
    f = (lambda a, b: f'{a:.1%} ± {b:.1%}') if pct else (lambda a, b: f'{a:.3f} ± {b:.3f}')
    return pd.DataFrame([[f(m.iloc[i, j], s.iloc[i, j]) for j in range(m.shape[1])] for i in range(m.shape[0])],
                        index=m.index, columns=m.columns)


def paired(vals):
    """Mean difference over seeds with a 95% t-interval, and in how many seeds it is positive.

    Seeds re-split the same 920 patients and re-randomise training, so the interval reflects
    split-and-training randomness, NOT sampling of new patients.
    """
    v = np.asarray(vals, float)
    n = len(v)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return dict(mean=m, lo=m - h, hi=m + h, positive=f'{int((v > 0).sum())}/{n}')
