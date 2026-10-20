"""Deliverables 2 and 3.

2. Earth-vs-Moon temperature table: for every returned soil, the temperature at which the
   melt reaches log eta = 3 (working point), 6.6 (softening) and 12 (Tg), for
   Earth-test conditions (air) and lunar conditions (IW-1), and the shift dT = T_moon - T_earth.
   Redox for both: Borisov 2018 at 1500 °C (Stapperfend's convention for the Earth side),
   giving Fe3+/Fe_tot ~0.6 (air) vs ~0.02 (IW-1). The glass keeps that redox on cooling.
   Error bars: deep-ensemble member spread (epistemic), as 95% intervals (+-1.96 sd).
   ensemble dT is computed member-by-member, so shared biases cancel.

3. Experiment ranking: composition x fO2 conditions ordered by ensemble disagreement
   (member sd of the iso-viscosity temperature). One representative per mission x terrane
   cluster so the list is not ten near-identical Apollo 16 soils.

Outputs: results/tables/earth_vs_moon_*.csv, results/tables/experiment_ranking.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import chem
import zoo

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
RES = ROOT / "results/tables"
T_REDOX = 1773.15
FE3_SAT = 0.5  # below this Fe3+/FeT, Chevrel's AD melts show no further Tg/viscosity change
LEVELS = {"working (10^3)": 3.0, "softening (10^6.6)": 6.6, "Tg (10^12)": 12.0}
FO2 = {"air": lambda T: chem.LOG_FO2_AIR, "IW+2": lambda T: chem.log_fo2_iw(T) + 2,
       "IW": lambda T: chem.log_fo2_iw(T), "IW-1": lambda T: chem.log_fo2_iw(T) - 1}


def redox_variant(w_feot_row, fo2_key):
    w = pd.DataFrame([w_feot_row])
    ws, r = chem.set_redox(w, FO2[fo2_key](T_REDOX), T_REDOX, model="B2018")
    return ws, float(r[0])


def fit_anchor():
    """Measurement-anchored redox sensitivity, scaled by iron content.

    From the titrated Chevrel series: slope_i = k * FeOT_i (through the origin), weighted by
    1/se^2 with floors (HT 0.03 log units, Tg 8 K per unit Fe3+/FeT) so 3-point series are not
    over-trusted. Returns {regime: (k, se_k)} with
      HT: d log eta / d(Fe3+/FeT) per wt% FeOT, measured at log eta ~ 0-0.6
      Tg: dT(iso-viscosity ~10^11) / d(Fe3+/FeT) per wt% FeOT
    """
    rs = pd.read_csv(P / "redox_series.csv")
    rs = rs[rs.quality != "low"]
    out, rows = {}, []
    for regime, ycol, floor in [("HT", "log_eta", 0.03), ("Tg", "Tf_K", 8.0)]:
        S, SE, F = [], [], []
        for series, g in rs[rs.regime == regime].groupby("series"):
            x, y = g.fe3_ratio.values, g[ycol].values
            b, a = np.polyfit(x, y, 1)
            se = np.sqrt(np.var(y - (a + b * x), ddof=2) / ((x - x.mean()) ** 2).sum()) if len(x) > 2 else np.inf
            se = max(se, floor)
            S.append(b), SE.append(se), F.append(g.feot.iloc[0])
            rows.append(dict(regime=regime, series=series, feot=g.feot.iloc[0], slope=b, se=se))
        S, SE, F = map(np.asarray, (S, SE, F))
        w = 1 / SE ** 2
        k = np.sum(w * F * S) / np.sum(w * F ** 2)
        resid = S - k * F
        # inflate by reduced chi^2 when the series scatter more than their own errors
        chi2 = np.sum(w * resid ** 2) / (len(S) - 1)
        se_k = np.sqrt(max(chi2, 1.0) / np.sum(w * F ** 2))
        out[regime] = (k, se_k)
    pd.DataFrame(rows).to_csv(RES / "redox_anchor_series.csv", index=False)
    print("anchored sensitivity per wt% FeOT:",
          {r: (round(k, 4), round(s, 4)) for r, (k, s) in out.items()})
    return out


def anchored_shift(anchor, feot, dfe3, A, B, C, T_earth_level, level):
    """dT (K) Moon - Earth at an iso-viscosity level from the measured redox sensitivity.
    HT anchor converts d log eta to dT with the local VFT slope at the level; the Tg anchor is
    already a temperature shift. Intermediate levels interpolate linearly in log eta between
    the HT anchor (log eta ~ 0.3) and the Tg anchor (log eta ~ 11). Returns (dT, se)."""
    k_ht, s_ht = anchor["HT"]
    k_tg, s_tg = anchor["Tg"]
    dlog = k_ht * feot * dfe3                      # change in log eta at fixed T
    dlog_se = s_ht * feot * abs(dfe3)
    slope_T = B / (T_earth_level - C) ** 2         # -d log eta / dT at this level (>0)
    dT_ht, se_ht = dlog / slope_T, dlog_se / slope_T
    dT_tg, se_tg = k_tg * feot * dfe3, s_tg * feot * abs(dfe3)
    f = np.clip((level - 0.3) / (11.0 - 0.3), 0, 1)
    return (1 - f) * dT_ht + f * dT_tg, (1 - f) * se_ht + f * se_tg


def member_T(ens, ws, target):
    """Per-member iso-viscosity temperature (analytic for VFT: T = C + B/(target - A))."""
    out = []
    for A, B, C, _ in ens.vft_params(ws):
        out.append(C[0] + B[0] / (target - A[0]))
    return np.array(out)


def table(fitted):
    soils = pd.read_csv(P / "lunar_soils.csv")
    zm = zoo.zoo(fitted)
    ens = fitted["ensemble"]
    anchor = fit_anchor()
    rows = []
    for _, s in soils.iterrows():
        w = s[chem.OXIDES_FET].astype(float)
        ws_e, r_e = redox_variant(w, "air")
        ws_m, r_m = redox_variant(w, "IW-1")
        _, r_e1100 = chem.set_redox(pd.DataFrame([w]), chem.LOG_FO2_AIR, 1373.15, model="B2018")
        r_e1100 = float(r_e1100[0])
        curves = {name: (np.asarray(zm[name](ws_e, zoo.T_GRID)[0], float),
                         np.asarray(zm[name](ws_m, zoo.T_GRID)[0], float))
                  for name in ["gpvisc 0.3.5", "Gradient boosting", "Ridge (NBO/T)", "GRD 2008"]}
        A_, B_, C_ = (np.mean([p[i][0] for p in ens.vft_params(ws_e)]) for i in range(3))
        for lvl, tgt in LEVELS.items():
            Te, Tm = member_T(ens, ws_e, tgt), member_T(ens, ws_m, tgt)
            d = Tm - Te
            dT_anc, se_anc = anchored_shift(anchor, s.feot, r_m - r_e, A_, B_, C_, Te.mean(), tgt)
            # saturating form: no further change once Fe3+/FeT < 0.5 (Chevrel AD series)
            dT_sat, se_sat = anchored_shift(anchor, s.feot, max(r_m, FE3_SAT) - r_e,
                                            A_, B_, C_, Te.mean(), tgt)
            # Earth reference sensitivity: powder sintered in air at ~1100 °C
            dT_lin_1100, _ = anchored_shift(anchor, s.feot, r_m - r_e1100, A_, B_, C_, Te.mean(), tgt)
            dT_sat_1100, _ = anchored_shift(anchor, s.feot, max(r_m, FE3_SAT) - r_e1100,
                                            A_, B_, C_, Te.mean(), tgt)
            row = dict(sample=s["sample"], mission=s.mission, terrane=s.terrane, FeOT=s.feot,
                       fe3_earth=r_e, fe3_moon=r_m, level=lvl,
                       T_earth=Te.mean(), T_earth_sd=Te.std(), T_moon=Tm.mean(), T_moon_sd=Tm.std(),
                       dT_anchored=dT_anc, dT_anc_lo=dT_anc - 1.96 * se_anc,
                       dT_anc_hi=dT_anc + 1.96 * se_anc,
                       dT_anchored_sat=dT_sat, dT_sat_lo=dT_sat - 1.96 * se_sat,
                       dT_sat_hi=dT_sat + 1.96 * se_sat,
                       fe3_earth_1100C=r_e1100, dT_lin_earth1100=dT_lin_1100,
                       dT_sat_earth1100=dT_sat_1100,
                       # same shift expressed as a viscosity change at a fixed process temperature
                       dlogeta_fixedT_anchored=-dT_anc * B_ / (Te.mean() - C_) ** 2,
                       dT_ensemble=d.mean(), dT_ens_sd=d.std(),
                       dT_ens_lo=d.mean() - 1.96 * d.std(), dT_ens_hi=d.mean() + 1.96 * d.std())
            for name in ["gpvisc 0.3.5", "Gradient boosting", "Ridge (NBO/T)", "GRD 2008"]:
                te, tm = (zoo.T_from_curve(curves[name][0], tgt), zoo.T_from_curve(curves[name][1], tgt))
                row[f"dT_{name}"] = tm - te
                row[f"T_earth_{name}"] = te
            rows.append(row)
        print(s["sample"], end=" ", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "earth_vs_moon_all.csv", index=False)
    # resolvability: is the ensemble shift distinguishable from zero at 95%?
    out["resolved"] = (out.dT_ens_hi < 0) | (out.dT_ens_lo > 0)
    out["anchored_resolved"] = (out.dT_anc_hi < 0) | (out.dT_anc_lo > 0)
    # models "too uncertain to say": ensemble and measurement anchor disagree beyond both bands
    lo = np.minimum(out.dT_anc_lo, out.dT_sat_lo)
    hi = np.maximum(out.dT_anc_hi, out.dT_sat_hi)
    out["model_vs_anchor_conflict"] = (out.dT_ens_lo > hi) | (out.dT_ens_hi < lo)
    out.to_csv(RES / "earth_vs_moon_all.csv", index=False)
    summ = out.groupby(["level", "terrane"]).agg(
        n=("sample", "size"), FeOT=("FeOT", "mean"),
        dT_anchored=("dT_anchored", "mean"), anc_lo=("dT_anc_lo", "mean"), anc_hi=("dT_anc_hi", "mean"),
        dlogeta_fixedT=("dlogeta_fixedT_anchored", "mean"),
        dT_sat=("dT_anchored_sat", "mean"), sat_lo=("dT_sat_lo", "mean"), sat_hi=("dT_sat_hi", "mean"),
        fe3_earth_1100C=("fe3_earth_1100C", "mean"),
        dT_lin_earth1100=("dT_lin_earth1100", "mean"), dT_sat_earth1100=("dT_sat_earth1100", "mean"),
        frac_conflict=("model_vs_anchor_conflict", "mean"), dT_mean=("dT_ensemble", "mean"),
        dT_min=("dT_ensemble", "min"), dT_max=("dT_ensemble", "max"),
        ci_halfwidth=("dT_ens_sd", lambda x: 1.96 * x.mean()), frac_resolved=("resolved", "mean"),
        gpvisc_dT=("dT_gpvisc 0.3.5", "mean"), gbm_dT=("dT_Gradient boosting", "mean"),
        T_earth=("T_earth", "mean"), T_earth_sd=("T_earth_sd", "mean")).reset_index()
    summ.to_csv(RES / "earth_vs_moon_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n", summ.round(1).to_string(index=False))
    return out


def ranking(fitted, top=25):
    soils = pd.read_csv(P / "lunar_soils.csv")
    ens = fitted["ensemble"]
    # one representative per mission x terrane: the soil closest to the cluster's mean FeOT
    reps = (soils.assign(dev=lambda d: (d.feot - d.groupby(["mission", "terrane"]).feot.transform("mean")).abs())
            .sort_values("dev").groupby(["mission", "terrane"]).head(1))
    sim = pd.read_csv(P / "stapperfend_compositions.csv")
    sim["mission"], sim["terrane"] = "simulant", "simulant"
    sim = sim.rename(columns={"paper": "ref"})
    cand = pd.concat([reps, sim], ignore_index=True)
    rows = []
    for _, s in cand.iterrows():
        w = s[chem.OXIDES_FET].astype(float)
        for fk in FO2:
            ws, r = redox_variant(w, fk)
            for lvl, tgt in LEVELS.items():
                T = member_T(ens, ws, tgt)
                mus, _ = ens.member_predictions(ws, np.array([T.mean()]))
                rows.append(dict(composition=s["sample"], mission=s.mission, terrane=s.terrane,
                                 FeOT=s.feot, fO2=fk, fe3_ratio=r, level=lvl,
                                 T_pred_K=T.mean(), disagreement_K=T.std(),
                                 disagreement_logeta=float(mus[:, 0].std())))
    r = pd.DataFrame(rows).sort_values("disagreement_logeta", ascending=False).reset_index(drop=True)
    r.insert(0, "rank", np.arange(1, len(r) + 1))
    r.to_csv(RES / "experiment_ranking_full.csv", index=False)
    # headline list: best condition per composition x fO2 (avoid repeating the same melt)
    # headline: the most informative condition per composition (one row per melt)
    head = r.drop_duplicates(["composition"]).head(top).copy()
    head["rank"] = np.arange(1, len(head) + 1)
    head.to_csv(RES / "experiment_ranking.csv", index=False)
    print(head.round(3).to_string(index=False))
    return r


def deliverable_table():
    """One row per soil: Earth-test temperature and Moon - Earth shift at each level."""
    d = pd.read_csv(RES / "earth_vs_moon_all.csv")
    short = {"working (10^3)": "work", "softening (10^6.6)": "soft", "Tg (10^12)": "Tg"}
    rows = []
    for s, g in d.groupby("sample", sort=False):
        r = dict(sample=s, mission=g.mission.iloc[0], terrane=g.terrane.iloc[0],
                 FeOT_wt=round(g.FeOT.iloc[0], 1))
        for _, x in g.iterrows():
            k = short[x.level]
            r[f"{k}_T_earth_C"] = round(x.T_earth - 273.15)
            r[f"{k}_dT_anchored_range_K"] = f"{x.dT_anchored_sat:.0f} to {x.dT_anchored:.0f}"
            r[f"{k}_dT_anchored_95ci_K"] = (f"[{min(x.dT_anc_lo, x.dT_sat_lo):.0f}, "
                                            f"{max(x.dT_anc_hi, x.dT_sat_hi):.0f}]")
            r[f"{k}_dT_ensemble_K"] = round(x.dT_ensemble, 1)
            r[f"{k}_dT_ensemble_95ci_K"] = f"[{x.dT_ens_lo:.0f}, {x.dT_ens_hi:.0f}]"
            r[f"{k}_dT_gpvisc_K"] = round(x["dT_gpvisc 0.3.5"], 1)
            r[f"{k}_model_conflicts_with_measurement"] = bool(x.model_vs_anchor_conflict)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("FeOT_wt")
    t.to_csv(RES / "DELIVERABLE2_earth_vs_moon_by_soil.csv", index=False)
    return t


if __name__ == "__main__":
    fitted = zoo.load_fitted()
    table(fitted)
    deliverable_table()
    ranking(fitted) if "--no-rank" not in __import__("sys").argv else None
