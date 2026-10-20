"""Step 4: redox stress test (the key result).

Test A - measured redox series (Chevrel 2013/2014, Fe2+/Fe_tot by titration; both studies
held out of our training data; pretrained gpvisc HAS seen them):
  HT: isothermal log eta vs Fe3+/Fe_tot  -> measured slope d log eta / d(Fe3+/Fe_tot)
  Tg: Tf'(10 K/min) vs Fe3+/Fe_tot      -> measured slope dT / d(Fe3+/Fe_tot)
  For each model we compute the same slopes from predictions (Tg regime: temperature at
  iso-viscosity log eta = 11, roughly where Tf' at 10 K/min sits for these melts).
  Redox-sensitivity error = predicted slope - measured slope.

Test B - train-oxidised / test-reduced: retrain with every Fe-rich (FeOT>=5 wt%) row having
  Fe3+/Fe_tot < 0.3 removed, then score on exactly those rows. Compare with a grouped random
  hold-out of the same size to separate "reduced is hard" from "any hold-out is hard".

Outputs: results/tables/redox_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import chem
import models
import zoo

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
RES = ROOT / "results/tables"
ISO_TG = 11.0


def series_wt(row_like, fe3):
    w = pd.DataFrame([row_like[chem.OXIDES_FET].astype(float)])
    return chem.split_iron(w, np.array([fe3]))


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) < 0.05:
        return np.nan
    return np.polyfit(x[ok], y[ok], 1)[0]


def test_a(zm):
    rs = pd.read_csv(P / "redox_series.csv")
    rows, pts = [], []
    for (series, regime), g in rs.groupby(["series", "regime"]):
        g = g[g.quality != "low"]
        base = g.iloc[0]
        if regime == "HT":
            meas = slope(g.fe3_ratio, g.log_eta)
            meas_span = g.loc[g.fe3_ratio.idxmax(), "log_eta"] - g.loc[g.fe3_ratio.idxmin(), "log_eta"]
        else:
            meas = slope(g.fe3_ratio, g.Tf_K)
            meas_span = g.loc[g.fe3_ratio.idxmax(), "Tf_K"] - g.loc[g.fe3_ratio.idxmin(), "Tf_K"]
        for name, f in zm.items():
            preds = []
            for _, r in g.iterrows():
                ws = series_wt(base, r.fe3_ratio)
                if regime == "HT":
                    mu, sd = f(ws, np.array([r.T_K]))
                    preds.append(float(np.asarray(mu)[0]))
                    pts.append(dict(series=series, regime=regime, model=name, fe3=r.fe3_ratio,
                                    obs=r.log_eta, pred=preds[-1], sd=float(np.asarray(sd)[0])))
                else:
                    preds.append(zoo.T_at(f, ws, ISO_TG))
                    pts.append(dict(series=series, regime=regime, model=name, fe3=r.fe3_ratio,
                                    obs=r.Tf_K, pred=preds[-1], sd=np.nan))
            pred = slope(g.fe3_ratio, preds)
            rows.append(dict(series=series, regime=regime, model=name, n=len(g),
                             feot=base.feot, fe3_lo=g.fe3_ratio.min(), fe3_hi=g.fe3_ratio.max(),
                             measured_slope=meas, predicted_slope=pred,
                             slope_error=pred - meas, measured_span=meas_span))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "redox_seriesA_slopes.csv", index=False)
    pd.DataFrame(pts).to_csv(RES / "redox_seriesA_points.csv", index=False)
    pd.set_option("display.width", 250)
    for regime, unit in [("HT", "log units per unit Fe3+/FeT"), ("Tg", "K per unit Fe3+/FeT")]:
        t = out[out.regime == regime].pivot(index="series", columns="model", values="predicted_slope")
        t.insert(0, "MEASURED", out[out.regime == regime].groupby("series").measured_slope.first())
        print(f"\n{regime} regime slopes ({unit}):\n", t.round(2).to_string())
    summ = out.groupby(["regime", "model"]).agg(
        mae_slope=("slope_error", lambda e: np.nanmean(np.abs(e))),
        sign_ok=("slope_error", lambda e: np.nan)).reset_index()
    sign = out.assign(ok=np.sign(out.predicted_slope) == np.sign(out.measured_slope)) \
              .groupby(["regime", "model"]).ok.mean().reset_index()
    summ = summ.drop(columns="sign_ok").merge(sign, on=["regime", "model"])
    summ.to_csv(RES / "redox_seriesA_summary.csv", index=False)
    print("\n", summ.round(2).to_string(index=False))
    return out


def test_b(k_ens=5):
    d = pd.read_csv(P / "train_lp.csv")
    d = d[~d.is_redox_study].reset_index(drop=True)
    red_mask = (d.feot >= 5) & (d.fe3_ratio < 0.3)
    reduced = d[red_mask]
    rng = np.random.default_rng(0)
    # grouped random control: pick whole groups among Fe-rich oxidised rows until size matches
    ox_groups = d[(d.feot >= 5) & ~red_mask].group.unique()
    rng.shuffle(ox_groups)
    ctrl_groups, n = [], 0
    for gid in ox_groups:
        if n >= len(reduced):
            break
        ctrl_groups.append(gid)
        n += (d.group == gid).sum()
    ctrl_mask = d.group.isin(ctrl_groups) & ~red_mask
    rows = []
    for split, test_mask in [("train-oxidised / test-reduced", red_mask),
                             ("control: grouped random Fe-rich hold-out", ctrl_mask)]:
        train = d[~test_mask & ~(d.group.isin(d[test_mask].group.unique()))]
        test = d[test_mask]
        for name, m in [("Ridge (NBO/T)", models.RidgePhys()), ("Gradient boosting", models.GBM()),
                        ("Deep ensemble", models.Ensemble(k=k_ens))]:
            m.fit(train)
            mu, sd = m.predict(test[chem.OXIDES], test.T_K)
            err = mu - test.log_eta.values
            rows.append(dict(split=split, model=name, n_train=len(train), n_test=len(test),
                             test_groups=test.group.nunique(), rmse=np.sqrt(np.mean(err ** 2)),
                             bias=err.mean(), cov95=np.mean(np.abs(err) < 1.96 * sd)))
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "redox_testB_split.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    stages = sys.argv[1:] or ["a", "b"]
    if "a" in stages:
        test_a(zoo.zoo(zoo.load_fitted()))
    if "b" in stages:
        test_b()
