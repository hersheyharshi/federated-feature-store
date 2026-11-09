"""Differential-privacy accounting for DP-SGD (Poisson-sampled Gaussian mechanism).

Renyi-DP accountant (Mironov et al. 2019, integer orders) with the RDP -> (eps, delta)
conversion of Balle et al. 2020 (same as Opacus). Guarantee: add/remove-one-record.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = np.arange(2, 257)


@dataclass(frozen=True)
class DPConfig:
    clip: float
    noise_multiplier: float
    delta: float = 1e-5


def _rdp_one_step(q, sigma, alpha):
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    k = np.arange(alpha + 1)
    log_binom = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    terms = log_binom + (alpha - k) * np.log1p(-q) + k * np.log(q) + (k ** 2 - k) / (2 * sigma ** 2)
    return float(logsumexp(terms)) / (alpha - 1)


def epsilon(q, sigma, steps, delta, orders=ORDERS):
    """Total epsilon after `steps` DP-SGD steps with sampling rate q."""
    rdp = np.array([steps * _rdp_one_step(q, sigma, int(a)) for a in orders])
    eps = rdp + np.log((orders - 1) / orders) - (np.log(delta) + np.log(orders)) / (orders - 1)
    return float(max(np.min(eps), 0.0))


def noise_for_epsilon(target, q, steps, delta, lo=0.3, hi=500.0):
    """Smallest noise multiplier whose total epsilon is <= target (binary search)."""
    if epsilon(q, hi, steps, delta) > target:
        raise ValueError('target epsilon unreachable even with huge noise')
    for _ in range(40):
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta) > target:
            lo = mid
        else:
            hi = mid
    return hi
