"""Step 2: reproduce Stapperfend 2025 baseline numbers to validate the pipeline.

Checks
  (a) GRD 2008 VFT parameters B, C per simulant vs AmMin Table 5 / CG Table 7.
  (b) GRD and gpvisc RMSE (superliquidus) and dEta(Tg) vs AmMin Table 6 / CG Table 6.

Their conventions, which we copy here:
  - redox: Kress & Carmichael 1991 at air (fO2 = 0.21), evaluated at 1500 °C
    (reproduces their printed Fe3+/Fe_tot = 0.59-0.62 to +-0.01).
  - AmMin RMSE uses only the 7 highest-T points (1344-1490 °C).
  - dEta(Tg) = measured - model log eta at Tg,DSC (positive: model too fluid at Tg).

Output: results/tables/reproduction_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import chem

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"

PUBLISHED_GRD = {  # sample: (B, C)
    "TUBS-M": (5719, 587), "TUBS-T": (5361, 672), "TUBS-I": (5568, 630),
    "Station-5": (5781, 677), "LRV-6": (5513, 670), "LRV-7/8": (5722, 633),
    "LX-M100": (5767, 587), "LX-T100": (4957, 695), "LMS-1": (4979, 632), "LHS-1": (5352, 662),
    "JSC-1A": (4894, 635), "EAC-1A": (4507, 649), "OPRL2N": (5026, 637), "OPRL2NT": (5145, 716),
    "OPRH2N": (4847, 679), "OPRH3N": (4788, 688),
}
# AmMin Table 6 (RMSE, dEta(Tg)) for GRD; gpvisc was not tested in AmMin.
PUBLISHED_AMMIN_GRD = {"TUBS-M": (0.071, -0.13), "TUBS-T": (0.853, 2.63), "TUBS-I": (0.366, -0.37),
                       "Station-5": (0.893, None), "LRV-6": (0.160, -1.44), "LRV-7/8": (0.364, -0.77)}


def simulant_inputs(redox_model="KC1991", T_redox=1773.15, log_fo2=chem.LOG_FO2_AIR):
    comp = pd.read_csv(P / "stapperfend_compositions.csv").set_index("sample")
    w = comp[chem.OXIDES_FET]
    w_split, r = chem.set_redox(w, log_fo2, T_redox, model=redox_model)
    return comp, w, w_split, pd.Series(r, index=w.index, name="fe3_ratio")


def main():
    comp, w, w_split, r = simulant_inputs()
    visc = pd.read_csv(P / "stapperfend_viscosity.csv")
    scores_cg = pd.read_csv(P / "stapperfend_model_scores_cg.csv")

    grd = baselines.grd2008_params(w)
    rows = []
    for s in w.index:
        d = visc[(visc["sample"] == s)]
        cc = d[(d.method == "CC") & d.valid]
        if comp.loc[s, "paper"] == "AmMin":
            cc = cc[cc.T_K >= 1344 + 273.15 - 0.5]
        tg = d[d.method == "DSC"]
        A, B, C = grd.loc[s]
        grd_rmse = np.sqrt(np.mean((baselines.vft(cc.T_K, A, B, C) - cc.log_eta) ** 2))
        grd_dtg = (tg.log_eta - baselines.vft(tg.T_K, A, B, C)).iloc[0] if len(tg) else np.nan

        ws = w_split.loc[[s] * len(cc)]
        gp_mean, _ = baselines.gpvisc_predict(ws, cc.T_K.values)
        gp_rmse = np.sqrt(np.mean((gp_mean - cc.log_eta.values) ** 2))
        if len(tg):
            gp_tg, _ = baselines.gpvisc_predict(w_split.loc[[s]], tg.T_K.values)
            gp_dtg = tg.log_eta.iloc[0] - gp_tg[0]
        else:
            gp_dtg = np.nan

        pb, pc = PUBLISHED_GRD[s]
        if comp.loc[s, "paper"] == "AmMin":
            p_rmse, p_dtg = PUBLISHED_AMMIN_GRD[s]
            gp_pub = (np.nan, np.nan)
        else:
            row = scores_cg[(scores_cg["sample"] == s) & (scores_cg.model == "GRD2008")].iloc[0]
            p_rmse, p_dtg = row.rmse_ht, row.deta_tg
            g = scores_cg[(scores_cg["sample"] == s) & (scores_cg.model == "gpvisc2025")].iloc[0]
            gp_pub = (g.rmse_ht, g.deta_tg)
        rows.append(dict(sample=s, paper=comp.loc[s, "paper"], fe3_ratio=r[s], n_ht=len(cc),
                         GRD_B=B, GRD_B_pub=pb, GRD_C=C, GRD_C_pub=pc,
                         GRD_rmse=grd_rmse, GRD_rmse_pub=p_rmse,
                         GRD_dTg=grd_dtg, GRD_dTg_pub=p_dtg,
                         gpvisc_rmse=gp_rmse, gpvisc_rmse_pub=gp_pub[0],
                         gpvisc_dTg=gp_dtg, gpvisc_dTg_pub=gp_pub[1]))
    out = pd.DataFrame(rows)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results/tables/reproduction_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    print("\nmax |dB| =", (out.GRD_B - out.GRD_B_pub).abs().max().round(1),
          " max |dC| =", (out.GRD_C - out.GRD_C_pub).abs().max().round(1))
    print("GRD rmse abs diff (median):", (out.GRD_rmse - out.GRD_rmse_pub).abs().median().round(3))
    print("gpvisc rmse abs diff (median, CG):",
          (out.gpvisc_rmse - out.gpvisc_rmse_pub).abs().median().round(3))


if __name__ == "__main__":
    main()
