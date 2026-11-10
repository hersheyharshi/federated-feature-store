"""Concept-drift simulation and detection."""
import numpy as np


def simulate_drift(values, y, rng, *, threshold, p_recover=0.9):
    """'A new treatment arrives': sick patients with a high risk marker now mostly recover.

    values: one risk score per patient (an input, signed so that higher = riskier).

    The inputs (and so P(x)) are unchanged; only P(y | x) changes - that is concept drift.
    Returns new labels and the fraction of labels that changed (the Delta used in the bound).
    """
    flag = (values > threshold) & (y == 1)
    recover = flag & (rng.random(len(y)) < p_recover)
    y_new = y.copy()
    y_new[recover] = 0
    return y_new, float((y_new != y).mean())


class DDM:
    """Drift Detection Method (Gama et al. 2004): alarm when the running error rate climbs
    well above its best level so far (3 standard deviations)."""

    def __init__(self, min_instances=30, warn=2.0, drift=3.0):
        self.min_instances, self.warn, self.drift = min_instances, warn, drift
        self.reset()

    def reset(self):
        self.n, self.p = 0, 1.0
        self.p_min, self.s_min = float('inf'), float('inf')

    def update(self, err):
        self.n += 1
        self.p += (err - self.p) / self.n
        s = np.sqrt(self.p * (1 - self.p) / self.n)
        if self.n < self.min_instances:
            return 'ok'
        if self.p + s <= self.p_min + self.s_min:
            self.p_min, self.s_min = self.p, s
        if self.p + s >= self.p_min + self.drift * self.s_min:
            return 'drift'
        if self.p + s >= self.p_min + self.warn * self.s_min:
            return 'warning'
        return 'ok'


def detect(errors, **kw):
    """Index of the first drift alarm in a 0/1 error stream, or None."""
    det = DDM(**kw)
    for i, e in enumerate(errors):
        if det.update(float(e)) == 'drift':
            return i
    return None
