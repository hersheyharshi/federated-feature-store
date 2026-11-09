"""The shared, versioned feature store.

One agreed schema (name, meaning, unit, valid range, version). Hospitals compute
SUMMARIES (count, sum, sum of squares) from their own records; only those travel.
The server combines them into shared means/spreads used to scale every input.
"""
import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

SCHEMA_VERSION = '1.0'


@dataclass(frozen=True)
class Feature:
    name: str
    meaning: str
    unit: str
    lo: float
    hi: float


# Ranges are deliberately generous: they only exist to catch impossible values
# (e.g. cholesterol 0 = "not recorded"). validation_report() shows what was removed.
SCHEMA = [
    Feature('age', 'Age of patient', 'years', 18, 100),
    Feature('sex', '1 = male, 0 = female', 'category', 0, 1),
    Feature('chest_pain', 'Chest pain type (1-4, 0 if coded so)', 'category', 0, 4),
    Feature('resting_bp', 'Resting blood pressure on admission', 'mm Hg', 50, 260),
    Feature('cholesterol', 'Serum cholesterol', 'mg/dl', 50, 700),
    Feature('fasting_sugar', 'Fasting blood sugar > 120 mg/dl', 'category', 0, 1),
    Feature('rest_ecg', 'Resting ECG result (0-2)', 'category', 0, 2),
    Feature('max_heart_rate', 'Maximum heart rate achieved', 'bpm', 40, 250),
    Feature('exercise_angina', 'Exercise-induced angina', 'category', 0, 1),
    Feature('st_depression', 'ST depression induced by exercise', 'mm', -3, 8),
    Feature('st_slope', 'Slope of peak exercise ST segment (0-3)', 'category', 0, 3),
    Feature('major_vessels', 'Major vessels coloured by fluoroscopy', 'count', 0, 4),
    Feature('thal', 'Thalassemia test result (3, 6, 7)', 'category', 0, 7),
]
NAMES = [f.name for f in SCHEMA]
LO = np.array([f.lo for f in SCHEMA], float)
HI = np.array([f.hi for f in SCHEMA], float)


def schema_hash():
    blob = json.dumps([asdict(f) for f in SCHEMA], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def clean(X):
    """Float array in schema order; anything outside the valid range becomes NaN."""
    A = X[NAMES].to_numpy(float).copy()
    A[(A < LO) | (A > HI)] = np.nan
    return A


def validation_report(frames):
    """How many values per hospital and feature fall outside the schema range."""
    rows = {}
    for h, df in frames.items():
        A = df[NAMES].to_numpy(float)
        rows[h] = {n: int(((A[:, j] < LO[j]) | (A[:, j] > HI[j])).sum()) for j, n in enumerate(NAMES)}
    return pd.DataFrame(rows).T


def missing_report(frames):
    return pd.DataFrame({h: pd.DataFrame(clean(df), columns=NAMES).isna().mean() for h, df in frames.items()}).T.round(2)


def local_stats(X):
    """What ONE hospital shares: per-feature (count, sum, sum of squares). No records."""
    A = clean(X)
    ok = ~np.isnan(A)
    V = np.where(ok, A, 0.0)
    return dict(n=ok.sum(0).astype(float), s=V.sum(0), ss=(V ** 2).sum(0))


def _laplace_noisy(stats, epsilon, rng):
    """Optional: Laplace noise on the shared sums. Sensitivities come from the schema range.

    Budget epsilon is split evenly over 3 queries x 13 features (basic composition, because
    one patient can touch every feature). Add/remove-one-patient neighbouring relation.
    """
    eps_q = epsilon / (3 * len(SCHEMA))
    sens_s = np.maximum(np.abs(LO), np.abs(HI))
    sens_ss = np.maximum(LO ** 2, HI ** 2)
    k = len(SCHEMA)
    return dict(n=np.maximum(stats['n'] + rng.laplace(0, 1 / eps_q, k), 0),
                s=stats['s'] + rng.laplace(0, sens_s / eps_q),
                ss=np.maximum(stats['ss'] + rng.laplace(0, sens_ss / eps_q), 0))


class FeatureStore:
    """Scaling parameters + schema version. transform() = clean -> standardise -> impute mean."""

    def __init__(self, mean, std, version=SCHEMA_VERSION, source='shared'):
        self.mean, self.std, self.version, self.source = mean, std, version, source

    @classmethod
    def from_stats(cls, stats_list, epsilon=None, rng=None):
        """Combine per-hospital summaries centrally (the 'shared averages and spreads')."""
        if epsilon is not None:
            rng = rng or np.random.default_rng(0)
            stats_list = [_laplace_noisy(s, epsilon, rng) for s in stats_list]
        n = sum(s['n'] for s in stats_list)
        s1 = sum(s['s'] for s in stats_list)
        s2 = sum(s['ss'] for s in stats_list)
        return cls._finish(n, s1, s2, 'shared' if epsilon is None else f'shared(eps={epsilon})')

    @classmethod
    def fit_local(cls, X):
        """What a hospital would do WITHOUT the store: scale with its own statistics only."""
        st = local_stats(X)
        return cls._finish(st['n'], st['s'], st['ss'], 'local')

    @classmethod
    def _finish(cls, n, s1, s2, source):
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(n > 0, s1 / np.maximum(n, 1e-9), 0.0)
            var = np.where(n > 1, s2 / np.maximum(n, 1e-9) - mean ** 2, 1.0)
        std = np.sqrt(np.maximum(var, 1e-6))
        std = np.where(std < 1e-3, 1.0, std)
        return cls(mean, std, SCHEMA_VERSION, source)

    def transform(self, X):
        A = (clean(X) - self.mean) / self.std
        return np.nan_to_num(A, nan=0.0)  # missing -> the mean (0 after standardising)

    def manifest(self):
        return dict(version=self.version, schema_hash=schema_hash(), source=self.source,
                    features=[dict(asdict(f), mean=float(m), std=float(s))
                              for f, m, s in zip(SCHEMA, self.mean, self.std)])
