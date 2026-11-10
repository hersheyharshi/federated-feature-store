"""Loading the four UCI heart-disease hospitals (and a synthetic stand-in for tests)."""
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 'fasting_sugar',
           'rest_ecg', 'max_heart_rate', 'exercise_angina', 'st_depression',
           'st_slope', 'major_vessels', 'thal', 'disease_level']
FEATURES = COLUMNS[:-1]
HOSPITAL_FILES = {
    'Cleveland': 'processed.cleveland.data',
    'Hungary': 'processed.hungarian.data',
    'Switzerland': 'processed.switzerland.data',
    'VA Long Beach': 'processed.va.data',
}
UCI_URL = 'https://archive.ics.uci.edu/static/public/45/heart+disease.zip'


def download_uci(raw_dir='data/raw'):
    """Download and unzip the UCI data once. Safe to call repeatedly."""
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists(os.path.join(raw_dir, HOSPITAL_FILES['Cleveland'])):
        return raw_dir
    zpath = os.path.join(raw_dir, 'heart.zip')
    urllib.request.urlretrieve(UCI_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw_dir)
    return raw_dir


def load_uci(raw_dir='data/raw'):
    """Return {hospital: DataFrame(13 features + disease_level + sick)}.

    '?' is read as missing. Range-based cleaning (e.g. cholesterol 0) is done
    by the feature store, not here, so the schema is the single source of truth.
    """
    frames = {}
    for name, file in HOSPITAL_FILES.items():
        df = pd.read_csv(os.path.join(raw_dir, file), header=None, names=COLUMNS, na_values='?')
        df['sick'] = (df['disease_level'] > 0).astype(int)
        frames[name] = df
    return frames


def synthetic_hospitals(seed=0):
    """Fake data with the SAME sizes / sick shares / missingness style as UCI.

    ONLY for tests and dry runs. Never report results from this.
    """
    rng = np.random.default_rng(seed)
    spec = {'Cleveland': (303, .46, 0.0, 0.0), 'Hungary': (294, .36, 0.3, 0.1),
            'Switzerland': (123, .93, 1.0, 0.2), 'VA Long Beach': (200, .74, 0.3, 0.2)}
    beta = rng.normal(0, 1, 13) * np.array([.6, .5, .8, .2, .2, .2, .2, -.7, .7, .8, .4, .8, .6])
    frames = {}
    for h, (n, share, chol_missing, other_missing) in spec.items():
        z = rng.normal(0, 1, (n, 13))
        z += rng.normal(0, .3, 13)  # hospital-specific shift
        raw = pd.DataFrame({
            'age': 54 + 9 * z[:, 0], 'sex': (z[:, 1] > -0.6).astype(float),
            'chest_pain': np.clip(np.round(3 + z[:, 2]), 1, 4),
            'resting_bp': 131 + 17 * z[:, 3], 'cholesterol': 240 + 50 * z[:, 4],
            'fasting_sugar': (z[:, 5] > 0.9).astype(float),
            'rest_ecg': np.clip(np.round(0.6 + 0.7 * z[:, 6]), 0, 2),
            'max_heart_rate': 150 + 22 * z[:, 7],
            'exercise_angina': (z[:, 8] > 0.4).astype(float),
            'st_depression': np.clip(1 + z[:, 9], -1, 6),
            'st_slope': np.clip(np.round(2 + 0.6 * z[:, 10]), 1, 3),
            'major_vessels': np.clip(np.round(0.6 + 0.8 * z[:, 11]), 0, 3),
            'thal': np.random.default_rng(1).choice([3., 6., 7.], n),
        })
        score = z @ beta
        thr = np.quantile(score, 1 - share)
        raw['disease_level'] = np.where(score + rng.normal(0, .8, n) > thr, rng.integers(1, 5, n), 0)
        for c in FEATURES:
            if c == 'cholesterol':
                miss = rng.random(n) < chol_missing
                raw.loc[miss, c] = 0.0  # UCI uses 0 for 'not recorded'
            elif rng.random() < 0.5:
                raw.loc[rng.random(n) < other_missing * 0.3, c] = np.nan
        raw['sick'] = (raw['disease_level'] > 0).astype(int)
        frames[h] = raw[COLUMNS + ['sick']]
    return frames


def split_hospitals(frames, test_size=0.30, seed=42):
    """70/30 practice/test split inside every hospital (stratified by label)."""
    out = {}
    for name, df in frames.items():
        try:
            Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df['sick'], test_size=test_size,
                                                  stratify=df['sick'], random_state=seed)
        except ValueError:  # a class too rare to stratify (small drift stages)
            Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df['sick'], test_size=test_size, random_state=seed)
        out[name] = dict(X_train=Xtr, y_train=ytr.to_numpy(), X_test=Xte, y_test=yte.to_numpy())
    return out
