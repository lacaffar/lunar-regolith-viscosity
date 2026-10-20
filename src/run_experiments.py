"""Step 3: train and score models.

1. Grouped 5-fold CV on the training set (groups = redox-invariant base composition).
2. Fit every model on the full training set (redox studies excluded) and save it.
3. Deliverable 1: scorecard on the 16 held-out Stapperfend simulants, against GRD 2008,
   pretrained gpvisc 0.3.5 and the published Sehlke & Whittington 2016 numbers.

Simulant redox for our models and gpvisc: Borisov 2018 at air, 1500 °C (Fe3+/Fe_tot ~0.58-0.65).

Outputs: results/models/*.pkl, results/tables/cv_scores.csv, results/tables/scorecard_*.csv
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import baselines
import chem
import models

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
RES = ROOT / "results"
(RES / "models").mkdir(parents=True, exist_ok=True)
(RES / "tables").mkdir(parents=True, exist_ok=True)

T_REDOX_EARTH = 1773.15  # K, Stapperfend's convention for their simulants (1500 °C)


def load_train():
    d = pd.read_csv(P / "train_lp.csv")
    return d[~d.is_redox_study].reset_index(drop=True)


def make_models():
    return {"ridge": models.RidgePhys(), "gbm": models.GBM(),
            "vft_mlp": models.VFTNet(seed=0), "ensemble": models.Ensemble(k=10)}


def _cv_fold(args):
    """One CV fold, single-threaded (folds run in parallel processes)."""
    import torch
    torch.set_num_threads(1)
    k, tr, te, d = args
    dt = d.iloc[te]
    rows, z = [], None
    for key, m in make_models().items():
        if key == "vft_mlp":
            continue  # member 0 of the ensemble is the single VFT-MLP
        t0 = time.time()
        m.fit(d.iloc[tr])
        preds = {key: m.predict(dt[chem.OXIDES], dt.T_K)}
        if key == "ensemble":
            preds["vft_mlp"] = m.members[0].predict(dt[chem.OXIDES], dt.T_K)
        for name, (mu, sd) in preds.items():
            err = mu - dt.log_eta.values
            if name == "ensemble":
                z = pd.DataFrame(dict(fold=k, z=err / sd, log_eta=dt.log_eta.values,
                                      fe_rich=dt.fe_rich.values))
            for regime, mask in [("all", np.ones(len(dt), bool)),
                                 ("HT (log eta<5)", dt.log_eta.values < 5),
                                 ("near Tg (log eta>=9)", dt.log_eta.values >= 9),
                                 ("Fe-rich (FeOT>=5)", dt.fe_rich.values)]:
                rows.append(dict(fold=k, model=name, regime=regime, n=int(mask.sum()),
                                 rmse=np.sqrt(np.mean(err[mask] ** 2)),
                                 cov95=np.mean(np.abs(err[mask]) < 1.96 * sd[mask])))
        print(f"fold {k} {key} done in {time.time() - t0:.0f}s", flush=True)
    return rows, z


def cross_validate(d):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(k, tr, te, d) for k, (tr, te) in enumerate(GroupKFold(5).split(d, groups=d.group))]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_cv_fold, jobs))
    cv = pd.DataFrame([r for rows, _ in results for r in rows])
    cv.to_csv(RES / "tables/cv_folds.csv", index=False)
    summ = cv.groupby(["model", "regime"]).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                                                cov95=("cov95", "mean"), n=("n", "sum")).reset_index()
    summ.to_csv(RES / "tables/cv_scores.csv", index=False)
    print(summ.round(3).to_string(index=False))
    # Ensemble sigma calibration from grouped-CV residuals (never touches the simulants):
    # scale so that 95% of |error| fall inside 1.96 * scale * sigma.
    z = pd.concat([zz for _, zz in results])
    z.to_csv(RES / "tables/cv_ensemble_z.csv", index=False)
    scale = float(np.quantile(np.abs(z.z), 0.95) / 1.96)
    (RES / "models/sigma_scale.txt").write_text(f"{scale:.4f}\n")
    print("ensemble sigma scale from CV:", round(scale, 3))
    return summ


def fit_full(d):
    fitted = {}
    for key, m in make_models().items():
        t0 = time.time()
        fitted[key] = m.fit(d)
        print(f"full fit {key}: {time.time() - t0:.0f}s", flush=True)
    with open(RES / "models/fitted.pkl", "wb") as f:
        pickle.dump(fitted, f)
    return fitted


def simulant_table():
    comp = pd.read_csv(P / "stapperfend_compositions.csv").set_index("sample")
    w = comp[chem.OXIDES_FET]
    wsplit, r = chem.set_redox(w, chem.LOG_FO2_AIR, T_REDOX_EARTH, model="B2018")
    visc = pd.read_csv(P / "stapperfend_viscosity.csv")
    visc = visc[visc.valid].reset_index(drop=True)
    return comp, w, wsplit, visc


def predictors(fitted):
    out = {
        "GRD 2008": lambda wt, wfeot, T: (baselines.vft(T, *baselines.grd2008_params(wfeot).iloc[0]),
                                          np.full(len(T), np.nan)),
        "gpvisc 0.3.5": lambda wt, wfeot, T: baselines.gpvisc_predict(wt.loc[[wt.index[0]] * len(T)], T),
    }
    names = {"ridge": "Ridge (NBO/T)", "gbm": "Gradient boosting", "vft_mlp": "VFT-MLP",
             "ensemble": "Deep ensemble"}
    for key, m in fitted.items():
        out[names[key]] = (lambda mm: lambda wt, wfeot, T:
                           mm.predict(wt.loc[[wt.index[0]] * len(T)], T))(m)
    return out


SW_PUBLISHED = {  # (RMSE_HT, dEta(Tg)) — AmMin Table 6 (7 hottest points) and CG Table 6
    "TUBS-M": (0.078, -0.41), "TUBS-T": (0.020, 1.98), "TUBS-I": (0.041, -0.42),
    "Station-5": (0.071, None), "LRV-6": (0.039, 1.00), "LRV-7/8": (0.061, 1.01),
    "LX-M100": (0.184, None), "LX-T100": (0.550, 1.56), "LMS-1": (0.089, None),
    "LHS-1": (0.164, 1.5), "JSC-1A": (0.108, -0.04), "EAC-1A": (0.191, None),
    "OPRL2N": (0.098, -0.21), "OPRL2NT": (0.122, None), "OPRH2N": (0.199, 0.57),
    "OPRH3N": (0.243, None),
}


def scorecard(fitted):
    comp, w, wsplit, visc = simulant_table()
    preds = predictors(fitted)
    rows, pts = [], []
    for s in w.index:
        v = visc[visc["sample"] == s]
        T = v.T_K.values
        for name, f in preds.items():
            mu, sd = f(wsplit.loc[[s]], w.loc[[s]], T)
            mu, sd = np.asarray(mu, float), np.asarray(sd, float)
            for i, (_, p) in enumerate(v.iterrows()):
                pts.append(dict(sample=s, model=name, method=p.method, T_K=p.T_K, obs=p.log_eta,
                                pred=mu[i], sd=sd[i]))
    pts = pd.DataFrame(pts)
    pts["err"] = pts.pred - pts.obs
    pts["in95"] = (np.abs(pts.err) < 1.96 * pts.sd).astype(float)
    pts.loc[pts.sd.isna(), "in95"] = np.nan
    pts.to_csv(RES / "tables/scorecard_points.csv", index=False)

    for (s, name), g in pts.groupby(["sample", "model"]):
        cc, dsc, fd = g[g.method == "CC"], g[g.method == "DSC"], g[g.method == "FDSC"]
        rows.append(dict(sample=s, model=name, rmse_ht=np.sqrt(np.mean(cc.err ** 2)),
                         deta_tg=(-dsc.err.iloc[0]) if len(dsc) else np.nan,  # measured - model
                         rmse_fdsc=np.sqrt(np.mean(fd.err ** 2)) if len(fd) else np.nan,
                         cov95=g.in95.mean() if g.in95.notna().any() else np.nan))
    for s, (r_, t_) in SW_PUBLISHED.items():
        rows.append(dict(sample=s, model="Sehlke & Whittington 2016 (published)", rmse_ht=r_,
                         deta_tg=t_, rmse_fdsc=0.194 if s == "OPRH2N" else np.nan, cov95=np.nan))
    per = pd.DataFrame(rows)
    per.to_csv(RES / "tables/scorecard_per_sample.csv", index=False)

    summ = per.groupby("model").agg(
        rmse_ht_mean=("rmse_ht", "mean"),
        rmse_ht_max=("rmse_ht", "max"),
        abs_deta_tg_mean=("deta_tg", lambda x: np.nanmean(np.abs(x))),
        abs_deta_tg_max=("deta_tg", lambda x: np.nanmax(np.abs(x))),
        n_tg_within_0p5=("deta_tg", lambda x: int((np.abs(x.dropna()) <= 0.5).sum())),
        n_tg=("deta_tg", lambda x: int(x.notna().sum())),
        cov95_mean=("cov95", "mean"),
    ).reset_index()
    pooled = pts.groupby("model").agg(rmse_all=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                                      cov95_all=("in95", "mean")).reset_index()
    summ = summ.merge(pooled, on="model", how="left").sort_values("rmse_ht_mean")
    summ.to_csv(RES / "tables/scorecard_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(per.pivot(index="sample", columns="model", values="deta_tg").round(2).to_string())
    return summ, per


if __name__ == "__main__":
    d = load_train()
    print("training rows", len(d), "groups", d.group.nunique(), flush=True)
    stages = sys.argv[1:] or ["cv", "fit", "score"]
    if "cv" in stages:
        cross_validate(d)
    if "fit" in stages:
        fitted = fit_full(d)
    else:
        import zoo
        fitted = zoo.load_fitted()  # applies the CV sigma calibration if it exists
    if "score" in stages:
        scorecard(fitted)
