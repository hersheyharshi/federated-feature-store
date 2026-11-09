import numpy as np
import pandas as pd

from fedfs import feature_store as fs
from fedfs.data import synthetic_hospitals, split_hospitals
from fedfs.drift import DDM, detect, simulate_drift
from fedfs.experiments import Config, run_baselines, run_epsilon_sweep, run_drift
from fedfs.federated import fedavg, local_train
from fedfs.models import add_bias, accuracy
from fedfs.privacy import epsilon, noise_for_epsilon

FRAMES = synthetic_hospitals(0)
FAST = Config(seeds=(0,), rounds=3, central_epochs=3, eps_grid=(1, 8), drift_rounds_a=3, drift_rounds_b=2)


def test_schema_cleans_impossible_values():
    X = FRAMES['Switzerland']
    assert np.isnan(fs.clean(X)[:, fs.NAMES.index('cholesterol')]).all()  # all zeros -> not recorded


def test_shared_stats_equal_pooled_stats():
    names = list(FRAMES)
    shared = fs.FeatureStore.from_stats([fs.local_stats(FRAMES[h]) for h in names])
    pooled = fs.clean(pd.concat(FRAMES.values()))
    assert np.allclose(shared.mean, np.nanmean(pooled, 0))
    assert np.allclose(shared.std, np.nanstd(pooled, 0), rtol=1e-3)


def test_local_stats_contain_no_records():
    st = fs.local_stats(FRAMES['Cleveland'])
    assert set(st) == {'n', 's', 'ss'} and all(v.shape == (13,) for v in st.values())


def test_privacy_accountant_sane():
    e1 = epsilon(1.0, 1.0, 1, 1e-5)
    assert 3 < e1 < 7                       # one full-batch Gaussian step with sigma=1
    assert epsilon(0.1, 1.0, 100, 1e-5) > epsilon(0.1, 1.0, 10, 1e-5)   # more steps cost more
    assert epsilon(0.1, 2.0, 100, 1e-5) < epsilon(0.1, 1.0, 100, 1e-5)  # more noise costs less
    assert epsilon(0.01, 1.0, 100, 1e-5) < epsilon(0.5, 1.0, 100, 1e-5)  # subsampling amplifies


def test_noise_calibration_hits_target():
    s = noise_for_epsilon(2.0, 0.2, 200, 1e-5)
    assert 1.9 < epsilon(0.2, s, 200, 1e-5) <= 2.0


def test_fedavg_learns_and_reports_epsilon():
    sp = split_hospitals(FRAMES)
    st = fs.FeatureStore.from_stats([fs.local_stats(sp[h]['X_train']) for h in sp])
    tr = {h: (add_bias(st.transform(sp[h]['X_train'])), sp[h]['y_train']) for h in sp}
    res = fedavg(tr, rounds=10, local_epochs=2, lr=.1, batch_size=32, seed=0, dp_epsilon=4.0)
    assert all(e <= 4.0 + 1e-6 for e in res.epsilons.values())
    plain = fedavg(tr, rounds=10, local_epochs=2, lr=.1, batch_size=32, seed=0)
    Xb, y = np.vstack([v[0] for v in tr.values()]), np.concatenate([v[1] for v in tr.values()])
    assert accuracy(plain.w, Xb, y) > 0.6


def test_dp_cannot_mix_with_replay():
    import pytest
    with pytest.raises(ValueError):
        local_train(np.zeros(3), np.ones((4, 3)), np.ones(4), epochs=1, lr=.1, batch_size=2,
                    rng=np.random.default_rng(0), dp=object(), replay=(np.ones((1, 3)), np.ones(1)))


def test_ddm_flags_a_jump_not_a_stable_stream():
    rng = np.random.default_rng(0)
    stable = (rng.random(300) < .2).astype(int)
    assert detect(stable) is None
    jump = np.concatenate([(rng.random(150) < .2), (rng.random(150) < .6)]).astype(int)
    a = detect(jump)
    assert a is not None and a >= 150


def test_simulated_drift_changes_only_sick_flagged_labels():
    rng = np.random.default_rng(0)
    v, y = rng.normal(size=500), rng.integers(0, 2, 500)
    y2, delta = simulate_drift(v, y, rng, threshold=0.0, p_recover=1.0)
    assert (y2 <= y).all() and delta == ((y2 != y).mean())
    assert (y2[v <= 0] == y[v <= 0]).all()


def test_experiments_smoke():
    b = run_baselines(FRAMES, FAST, 0)
    assert b['acc'].shape[1] == 5 and b['grid'].shape == (4, 4)
    sw = run_epsilon_sweep(FRAMES, FAST, 0)
    assert set(sw.epsilon) == {np.inf, 1, 8}
    d = run_drift(FRAMES, FAST, 0)
    assert len(d['summary']) == 1 + 1 + len(FAST.ewc_lams)


def test_accountant_matches_published_reference():
    """TF-Privacy MNIST example (N=60000, batch 256, sigma 1.1, 60 epochs, delta 1e-5) quotes eps = 3.01
    with the classic RDP conversion. Our RDP core must reproduce it; our default (tighter) conversion is lower."""
    from fedfs.privacy import _rdp_one_step, ORDERS
    q, T, d = 256 / 60000, int(60 * 60000 / 256), 1e-5
    rdp = np.array([T * _rdp_one_step(q, 1.1, int(a)) for a in ORDERS])
    classic = (rdp + np.log(1 / d) / (ORDERS - 1)).min()
    assert abs(classic - 3.01) < 0.02
    assert epsilon(q, 1.1, T, d) < classic


def test_mlp_gradients_match_numerical():
    from fedfs.models import get_model, sigmoid
    rng = np.random.default_rng(0)
    X = np.hstack([rng.normal(size=(7, 5)), np.ones((7, 1))])
    y = rng.integers(0, 2, 7).astype(float)
    for name in ('logistic', 'mlp8'):
        m = get_model(name)
        w = m.init(6, rng)
        w = w + rng.normal(0, .1, len(w))

        def loss(v):
            p = sigmoid(m.logits(v, X))
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        g = m.per_sample_grad(w, X, y)
        num = np.zeros_like(g)
        for k in range(len(w)):
            e = np.zeros(len(w)); e[k] = 1e-6
            num[:, k] = (loss(w + e) - loss(w - e)) / 2e-6
        assert np.abs(g - num).max() < 1e-6


def test_calibration_offset_matches_base_rate():
    from fedfs.models import fit_offset, sigmoid
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, 2000)
    y = (rng.random(2000) < sigmoid(logits + 1.5)).astype(float)
    assert abs(fit_offset(logits, y) - 1.5) < 0.25


def test_paired_interval_contains_true_shift():
    from fedfs.experiments import paired
    rng = np.random.default_rng(0)
    p = paired(0.5 + rng.normal(0, .1, 10))
    assert p['lo'] < 0.5 < p['hi'] and p['positive'] == '10/10'
