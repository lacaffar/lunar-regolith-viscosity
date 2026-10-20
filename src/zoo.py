"""Uniform prediction interface over every model, plus iso-viscosity temperature solver.

f(wt_split, T_K) -> (mean, sd), where wt_split is ONE composition (a 1-row DataFrame of wt%
with feo/fe2o3) and T_K an array. GRD sees only total iron (it is redox-blind by design).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import chem

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"ridge": "Ridge (NBO/T)", "gbm": "Gradient boosting", "vft_mlp": "VFT-MLP",
         "ensemble": "Deep ensemble"}


def to_feot(wt_split):
    w = wt_split.copy()
    w["feot"] = chem.feot_wt(w)
    return w[chem.OXIDES_FET]


def load_fitted(path=ROOT / "results/models/fitted.pkl", calibrated=True):
    with open(path, "rb") as f:
        fitted = pickle.load(f)
    sc = ROOT / "results/models/sigma_scale.txt"
    if calibrated and sc.exists():
        fitted["ensemble"].sigma_scale = float(sc.read_text())
    return fitted


def zoo(fitted, include_baselines=True):
    out = {}
    if include_baselines:
        def grd(ws, T):
            A, B, C = baselines.grd2008_params(to_feot(ws)).iloc[0]
            return baselines.vft(T, A, B, C), np.full(len(np.atleast_1d(T)), np.nan)

        def gpv(ws, T):
            T = np.atleast_1d(np.asarray(T, float))
            return baselines.gpvisc_predict(ws.loc[[ws.index[0]] * len(T)], T)
        out["GRD 2008"] = grd
        out["gpvisc 0.3.5"] = gpv
    for key, m in fitted.items():
        def f(ws, T, m=m):
            T = np.atleast_1d(np.asarray(T, float))
            return m.predict(ws.loc[[ws.index[0]] * len(T)], T)
        out[NAMES.get(key, key)] = f
    return out


T_GRID = np.arange(650.0, 2400.0, 2.0)


def T_at(f, ws, target, T_grid=T_GRID):
    """Lowest-viscosity-side crossing: scan from hot to cold, return T where log eta first
    reaches `target`. NaN if never reached on the grid."""
    mu, _ = f(ws, T_grid)
    return T_from_curve(np.asarray(mu, float), target, T_grid)


def T_from_curve(mu, target, T_grid=T_GRID):
    order = np.argsort(-T_grid)
    Tg, m = T_grid[order], mu[order]
    above = np.where(m >= target)[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan
    i = above[0]
    t0, t1, m0, m1 = Tg[i - 1], Tg[i], m[i - 1], m[i]
    return t0 + (target - m0) * (t1 - t0) / (m1 - m0)
