"""Run the whole study:  python -m fedfs.run_all --raw data/raw --out results --download
(add --synthetic for a dry run on fake data, --seeds N to change the number of seeds)"""
import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from . import feature_store as fs
from .data import load_uci, download_uci, synthetic_hospitals
from .experiments import (Config, run_baselines, run_epsilon_sweep, run_store_privacy, run_drift,
                          run_membership_test, mean_std, fmt, paired)
from .plots import plot_grid, plot_privacy_curve, plot_drift


def md(df, index=True):
    cols = ([df.index.name or ''] if index else []) + [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for i, r in df.iterrows():
        lines.append('| ' + ' | '.join(([str(i)] if index else []) + [str(v) for v in r.values]) + ' |')
    return '\n'.join(lines)


def ms(x, d=1):
    return f'{x:.{d}f}'


def run_model(frames, cfg, out, full):
    """Everything for one model. `full` adds the studies that are only run for the main model."""
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    names = list(frames)
    S, R = [], {}
    # --- baselines
    B = [run_baselines(frames, cfg, s) for s in cfg.seeds]
    gm, _ = mean_std([b['grid'] for b in B])
    gm.round(3).to_csv(os.path.join(out, 'local_grid.csv'))
    plot_grid(gm, os.path.join(out, 'figures', 'local_grid.png'))
    am, as_ = mean_std([b['acc'] for b in B])
    um, us = mean_std([b['auc'] for b in B])
    fmt(am, as_).to_csv(os.path.join(out, 'baseline_table.csv'))
    fmt(um, us, pct=False).to_csv(os.path.join(out, 'baseline_auc.csv'))
    S += ['### Test accuracy (mean ± std over seeds)\n', md(fmt(am, as_)),
          '\n### AUC (not fooled by the lazy-guess trap)\n', md(fmt(um, us, pct=False)),
          '\n### Local grid (rows trained at, columns tested at)\n', md(fmt(gm, mean_std([b["grid"] for b in B])[1])), '']
    # --- paired comparisons
    A, U = 'Federated, shared feature store', None
    comps = [
        ('Shared store vs own scaling', 'acc', A, 'Federated, own scaling', 'Average'),
        ('Federated (shared store) vs pooled ceiling', 'acc', A, 'Pooled (ceiling)', 'Average'),
        ('Federated (shared store) vs local at home', 'acc', A, 'Local, at home', 'Average'),
        ('Federated + calibration vs local at home', 'acc', 'Federated + store + local calibration', 'Local, at home', 'Average'),
        ('Federated + calibration vs federated, Switzerland', 'acc', 'Federated + store + local calibration', A, 'Switzerland'),
        ('Federated + calibration vs local at home, Switzerland', 'acc', 'Federated + store + local calibration', 'Local, at home', 'Switzerland'),
        ('Federated + calibration vs local at home, VA Long Beach', 'acc', 'Federated + store + local calibration', 'Local, at home', 'VA Long Beach'),
        ('Federated (shared store) vs local at home, AUC', 'auc', A, 'Local, at home', 'Average'),
        ('Federated (shared store) vs pooled, AUC', 'auc', A, 'Pooled (ceiling)', 'Average'),
    ]
    rows = []
    for label, key, a, b, col in comps:
        p = paired([x[key].loc[a, col] - x[key].loc[b, col] for x in B])
        rows.append({'comparison': label, 'mean difference (points / AUC)': ms(p['mean'] * (100 if key == 'acc' else 1), 1 if key == 'acc' else 3),
                     '95% CI': f"[{ms(p['lo'] * (100 if key == 'acc' else 1), 1 if key == 'acc' else 3)}, {ms(p['hi'] * (100 if key == 'acc' else 1), 1 if key == 'acc' else 3)}]",
                     'positive in seeds': p['positive']})
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(out, 'paired_comparisons.csv'), index=False)
    S += ['### Paired comparisons (difference per seed; 95% t-interval over seeds)\n',
          'Seeds re-split the same patients, so intervals reflect split and training randomness, not new patients.\n',
          md(pc, index=False), '']
    # --- epsilon sweep
    sweep = pd.concat([run_epsilon_sweep(frames, cfg, s) for s in cfg.seeds])
    sweep.to_csv(os.path.join(out, 'epsilon_sweep_raw.csv'), index=False)
    ps = sweep.groupby(['epsilon', 'seed'])[['accuracy', 'accuracy_cal']].mean().reset_index().groupby('epsilon')[['accuracy', 'accuracy_cal']].agg(['mean', 'std'])
    tbl = pd.DataFrame({'accuracy': [f'{ps.loc[e, ("accuracy", "mean")]:.1%} ± {ps.loc[e, ("accuracy", "std")]:.1%}' for e in ps.index],
                        'with local calibration': [f'{ps.loc[e, ("accuracy_cal", "mean")]:.1%} ± {ps.loc[e, ("accuracy_cal", "std")]:.1%}' for e in ps.index]},
                       index=pd.Index(ps.index, name='epsilon'))
    tbl.to_csv(os.path.join(out, 'epsilon_sweep.csv'))
    ref = {'Pooled ceiling': am.loc['Pooled (ceiling)', 'Average'], 'Local at home': am.loc['Local, at home', 'Average'],
           'Lazy guess': am.loc['Lazy guess', 'Average']}
    plot_privacy_curve(sweep, ref, os.path.join(out, 'figures', 'privacy_curve.png'))
    S += ['### Privacy versus accuracy (federated, shared store, delta=1e-5)\n', md(tbl), '']
    # paired: eps vs none
    per = sweep.groupby(['epsilon', 'seed'])['accuracy'].mean().unstack('epsilon')
    rows = []
    for e in cfg.eps_grid:
        p = paired(per[e] - per[np.inf])
        rows.append({'epsilon': e, 'change vs no privacy (points)': ms(p['mean'] * 100), '95% CI': f"[{ms(p['lo'] * 100)}, {ms(p['hi'] * 100)}]"})
    cost = pd.DataFrame(rows)
    cost.to_csv(os.path.join(out, 'privacy_cost_paired.csv'), index=False)
    S += ['### Cost of privacy, paired by seed\n', md(cost, index=False), '']
    # --- store privacy
    if full:
        sp_ = pd.concat([run_store_privacy(frames, cfg, s) for s in cfg.seeds])
        sp_.to_csv(os.path.join(out, 'store_privacy_raw.csv'), index=False)
        g = sp_.groupby(['store_epsilon', 'total_epsilon'])['accuracy'].agg(['mean', 'std']).reset_index()
        st = pd.DataFrame({'summary epsilon': g['store_epsilon'], 'total epsilon (training + summaries)': g['total_epsilon'],
                           'accuracy': [f'{m:.1%} ± {s:.1%}' for m, s in zip(g['mean'], g['std'])]})
        st.to_csv(os.path.join(out, 'store_privacy.csv'), index=False)
        S += [f'### Making the shared feature-store summaries private too (training epsilon fixed at {cfg.store_train_eps:g})\n',
              'summary epsilon = inf means exact (non-private) summaries.\n', md(st, index=False), '']
    # --- drift
    D = [run_drift(frames, cfg, s) for s in cfg.seeds]
    ds = pd.concat([d['summary'] for d in D])
    agg = ds.groupby('strategy', sort=False).agg(['mean', 'std']).drop(columns='seed')
    cols = ['acc_old_before', 'acc_new_before', 'acc_old_after', 'acc_new_after', 'forgetting', 'both_stages_after']
    dt = pd.DataFrame({c: [f'{agg.loc[s, (c, "mean")]:.1%} ± {agg.loc[s, (c, "std")]:.1%}' for s in agg.index] for c in cols}, index=agg.index)
    dt.to_csv(os.path.join(out, 'drift_forgetting.csv'))
    plot_drift(D[0]['curves'], D[0]['phase_a'], os.path.join(out, 'figures', 'drift_forgetting.png'))
    # paired: replay / EWC vs none on the mean of both stages and forgetting
    prow = []
    base = ds[ds.strategy == 'No protection'].set_index('seed')
    for strat in [s for s in agg.index if s != 'No protection']:
        cur = ds[ds.strategy == strat].set_index('seed')
        pf = paired(base['forgetting'] - cur['forgetting'])
        pb = paired(cur['both_stages_after'] - base['both_stages_after'])
        pn = paired(cur['acc_new_after'] - base['acc_new_after'])
        prow.append({'strategy vs no protection': strat,
                     'forgetting avoided (points)': f"{ms(pf['mean'] * 100)} [{ms(pf['lo'] * 100)}, {ms(pf['hi'] * 100)}]",
                     'new-stage accuracy change (points)': f"{ms(pn['mean'] * 100)} [{ms(pn['lo'] * 100)}, {ms(pn['hi'] * 100)}]",
                     'mean-of-both-stages change (points)': f"{ms(pb['mean'] * 100)} [{ms(pb['lo'] * 100)}, {ms(pb['hi'] * 100)}]"})
    pdrift = pd.DataFrame(prow)
    pdrift.to_csv(os.path.join(out, 'drift_paired.csv'), index=False)
    det = pd.DataFrame([d['detection'] for d in D])
    det.index.name = 'seed'
    det.to_csv(os.path.join(out, 'drift_detection.csv'))
    bnd = pd.DataFrame([d['bound'] for d in D]).round(3)
    bnd.to_csv(os.path.join(out, 'drift_bound_check.csv'), index=False)
    found = det['delay'].dropna()
    S += ['### Forgetting and recovery (p_recover = %.1f)\n' % cfg.drift_p_recover, md(dt), '\n### Paired against no protection (95% CI)\n',
          md(pdrift, index=False),
          f'\n### Drift alarm (DDM)\nDetected after the drift in {len(found)}/{len(det)} seeds'
          + (f' (median delay {found.median():.0f} patients, range {found.min():.0f} to {found.max():.0f})' if len(found) else '')
          + f'; false alarms before the drift: {int(det.false_alarm.sum())}/{len(det)}.\n',
          '### Drift bound check\nHeld in %d/%d seeds. Delta range %.2f to %.2f.\n' % (bnd.bound_holds.sum(), len(bnd), bnd.delta.min(), bnd.delta.max())]
    # --- severity sweep
    if full:
        rows = []
        for p in cfg.severities:
            c2 = replace(cfg, drift_p_recover=p)
            Dp = [run_drift(frames, c2, s) for s in cfg.seeds]
            sm = pd.concat([d['summary'] for d in Dp])
            dd = pd.DataFrame([d['detection'] for d in Dp])
            dl = dd['delay'].dropna()
            for strat in ['No protection', 'Replay', 'EWC (lambda=10)']:
                x = sm[sm.strategy == strat]
                rows.append({'drift strength (p_recover)': p, 'strategy': strat,
                             'old model on new stage': f"{x.acc_new_before.mean():.1%}",
                             'forgetting': f"{x.forgetting.mean():.1%} ± {x.forgetting.std():.1%}",
                             'new-stage accuracy after': f"{x.acc_new_after.mean():.1%}",
                             'alarm detected': f"{len(dl)}/{len(dd)}", 'median delay': f"{dl.median():.0f}" if len(dl) else 'n/a',
                             'false alarms': int(dd.false_alarm.sum())})
        sev = pd.DataFrame(rows)
        sev.to_csv(os.path.join(out, 'drift_severity.csv'), index=False)
        S += ['### Gentler and stronger drift\n', md(sev, index=False), '']
    # --- membership
    M = pd.concat([run_membership_test(frames, cfg, s) for s in cfg.seeds])
    M.to_csv(os.path.join(out, 'membership_raw.csv'), index=False)
    mt = M.groupby('model')[['attack_auc', 'train_acc', 'test_acc']].agg(['mean', 'std'])
    mtab = pd.DataFrame({'model': mt.index,
                         'attack AUC': [f'{mt.loc[m, ("attack_auc", "mean")]:.3f} ± {mt.loc[m, ("attack_auc", "std")]:.3f}' for m in mt.index],
                         'train accuracy': [f'{mt.loc[m, ("train_acc", "mean")]:.1%}' for m in mt.index],
                         'test accuracy': [f'{mt.loc[m, ("test_acc", "mean")]:.1%}' for m in mt.index]})
    mtab.to_csv(os.path.join(out, 'membership.csv'), index=False)
    S += ['### Membership-guessing attack (AUC 0.5 = attacker guesses; averaged over the 4 hospitals)\n', md(mtab, index=False), '']
    return '\n'.join(S)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--out', default='results')
    ap.add_argument('--synthetic', action='store_true', help='fake data, dry run only')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--models', default='logistic,mlp16')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    frames = synthetic_hospitals() if a.synthetic else (download_uci(a.raw) and load_uci(a.raw) if a.download else load_uci(a.raw))
    names = list(frames)
    tag = 'SYNTHETIC DRY RUN - NOT REAL RESULTS\n\n' if a.synthetic else ''
    seeds = tuple(range(a.seeds))
    store = fs.FeatureStore.from_stats([fs.local_stats(frames[h]) for h in names])
    with open(os.path.join(a.out, 'feature_store_v1.json'), 'w') as f:
        json.dump(store.manifest(), f, indent=2)
    S = [f'# Results summary\n\n{tag}Seeds: {list(seeds)}. Settings are fixed in `Config`. The neural-network settings were compared on two spare split seeds of the same patients (not an independent test).\n',
         '## Data checks\n', 'Values outside the schema range (count):\n', md(fs.validation_report(frames)),
         '\nShare of each input missing after cleaning:\n', md(fs.missing_report(frames)), '']
    for i, m in enumerate(a.models.split(',')):
        cfg = Config(seeds=seeds, model=m)
        title = 'Main model: logistic regression' if m == 'logistic' else f'Robustness: small neural network ({m})'
        S += [f'## {title}\n', run_model(frames, cfg, os.path.join(a.out, m), full=(i == 0))]
        print(f'finished {m}', file=sys.stderr)
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(S))
    print('\n'.join(S))


if __name__ == '__main__':
    sys.exit(main())
