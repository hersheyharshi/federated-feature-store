# src/fedfs

Reusable code. Pure NumPy / scikit-learn.

| File | What it does |
|---|---|
| `data.py` | Loads the four UCI hospitals; a synthetic stand-in for dry runs only |
| `feature_store.py` | Versioned schema, cleaning, shared summaries (count, sum, sum of squares), scaling |
| `models.py` | Logistic regression and a small neural network, one flat weight vector, per-patient gradients, Fisher, local calibration |
| `privacy.py` | RDP accountant for DP-SGD and noise calibration for a target epsilon |
| `federated.py` | Local training (DP, EWC, replay) and federated averaging |
| `drift.py` | Concept-drift simulation and the DDM detector |
| `experiments.py` | Baselines, epsilon sweep, drift/forgetting, membership test |
| `plots.py`, `run_all.py` | Figures and the one-command runner |

Run everything: `python -m fedfs.run_all --raw data/raw --out results --download`
Dry run on fake data (never report these numbers): add `--synthetic`.
