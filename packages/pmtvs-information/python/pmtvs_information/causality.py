"""Causality analysis primitives."""
import numpy as np
from typing import Tuple


def granger_causality(source, target, max_lag=5) -> Tuple[float, float]:
    """Granger causality test."""
    source = np.asarray(source, dtype=np.float64).flatten()
    target = np.asarray(target, dtype=np.float64).flatten()
    n = min(len(source), len(target))
    source, target = source[:n], target[:n]
    mask = ~(np.isnan(source) | np.isnan(target))
    source, target = source[mask], target[mask]
    n = len(source)
    if n < max_lag + 10:
        return np.nan, np.nan
    best_aic = np.inf
    best_lag = 1
    for lag in range(1, max_lag + 1):
        y = target[lag:]
        X_r = np.column_stack([target[lag - i - 1:n - i - 1] for i in range(lag)])
        X_r = np.column_stack([np.ones(len(y)), X_r])
        try:
            beta = np.linalg.lstsq(X_r, y, rcond=None)[0]
            resid = y - X_r @ beta
            ssr = np.sum(resid ** 2)
            aic = len(y) * np.log(ssr / len(y) + 1e-10) + 2 * X_r.shape[1]
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        except:
            continue
    lag = best_lag
    y = target[lag:]
    X_r = np.column_stack([np.ones(len(y))] + [target[lag - i - 1:n - i - 1] for i in range(lag)])
    X_u = np.column_stack([X_r] + [source[lag - i - 1:n - i - 1] for i in range(lag)])
    try:
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
        ssr_r = np.sum((y - X_r @ beta_r) ** 2)
        ssr_u = np.sum((y - X_u @ beta_u) ** 2)
        df1 = lag
        df2 = len(y) - X_u.shape[1]
        if df2 <= 0 or ssr_u <= 0:
            return np.nan, np.nan
        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        from scipy.stats import f as f_dist
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
        return float(f_stat), float(p_value)
    except:
        return np.nan, np.nan


def convergent_cross_mapping(sig_a, sig_b, embedding_dim=3, tau=1, lib_sizes=None) -> Tuple[float, float]:
    """Convergent cross mapping for nonlinear causality."""
    sig_a = np.asarray(sig_a, dtype=np.float64).flatten()
    sig_b = np.asarray(sig_b, dtype=np.float64).flatten()
    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]
    mask = ~(np.isnan(sig_a) | np.isnan(sig_b))
    sig_a, sig_b = sig_a[mask], sig_b[mask]
    n = len(sig_a)
    n_embed = n - (embedding_dim - 1) * tau
    if n_embed < embedding_dim + 2:
        return np.nan, np.nan

    def embed(s):
        ne = len(s) - (embedding_dim - 1) * tau
        emb = np.zeros((ne, embedding_dim))
        for d in range(embedding_dim):
            emb[:, d] = s[d * tau:d * tau + ne]
        return emb

    def cross_map_skill(emb_x, target_y):
        ne = len(emb_x)
        predicted = np.zeros(ne)
        for i in range(ne):
            dists = np.linalg.norm(emb_x - emb_x[i], axis=1)
            dists[i] = np.inf
            nn_idx = np.argsort(dists)[:embedding_dim + 1]
            nn_dists = dists[nn_idx]
            min_d = nn_dists[0]
            if min_d < 1e-10:
                weights = np.zeros(len(nn_idx))
                weights[0] = 1.0
            else:
                weights = np.exp(-nn_dists / min_d)
            weights /= weights.sum()
            predicted[i] = np.sum(weights * target_y[nn_idx])
        corr = np.corrcoef(predicted, target_y[:ne])[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    emb_a = embed(sig_a)
    emb_b = embed(sig_b)
    target_a = sig_a[:n_embed]
    target_b = sig_b[:n_embed]
    rho_a = cross_map_skill(emb_b, target_a)
    rho_b = cross_map_skill(emb_a, target_b)
    return float(rho_a), float(rho_b)


def phase_coupling(signal1, signal2) -> float:
    """Phase-locking value between two signals."""
    from scipy.signal import hilbert
    signal1 = np.asarray(signal1, dtype=np.float64).flatten()
    signal2 = np.asarray(signal2, dtype=np.float64).flatten()
    n = min(len(signal1), len(signal2))
    signal1, signal2 = signal1[:n], signal2[:n]
    mask = ~(np.isnan(signal1) | np.isnan(signal2))
    signal1, signal2 = signal1[mask], signal2[mask]
    if len(signal1) < 10:
        return np.nan
    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))
    phase_diff = phase1 - phase2
    plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
    return plv
