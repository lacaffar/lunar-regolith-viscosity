"""Check our SW2016 implementation against Stapperfend 2025's published SW numbers
(per-simulant Cp_conf, Tg*, superliquidus RMSE and dEta(Tg)). Writes
results/tables/sw2016_reproduction.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

import chem
import sw2016 as sw

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
PUB = {  # sample: (Cp_conf, Tg3R*, RMSE_HT, dEta(Tg)) from AmMin Tables 5-6 and CG Tables 6-7
    "TUBS-M": (22.8, 927, 0.078, -0.41), "TUBS-T": (19.7, 1053, 0.020, 1.98),
    "TUBS-I": (21.3, 987, 0.041, -0.42), "Station-5": (24.2, 900, 0.071, None),
    "LRV-6": (22.5, 958, 0.039, 1.00), "LRV-7/8": (22.9, 932, 0.061, 1.01),
    "LX-M100": (22.2, 946, 0.184, None), "LX-T100": (20.0, 1068, 0.550, 1.56),
    "LMS-1": (20.7, 930, 0.089, None), "LHS-1": (20.7, 1024, 0.164, 1.5),
    "JSC-1A": (24.3, 923, 0.108, -0.04), "EAC-1A": (27.1, 880, 0.191, None),
    "OPRL2N": (23.4, 940, 0.098, -0.21), "OPRL2NT": (25.4, 890, 0.122, None),
    "OPRH2N": (21.6, 1009, 0.199, 0.57), "OPRH3N": (21.2, 1023, 0.243, None),
}


def main():
    comp = pd.read_csv(P / "stapperfend_compositions.csv").set_index("sample")
    visc = pd.read_csv(P / "stapperfend_viscosity.csv")
    visc = visc[visc.valid]
    w = comp[chem.OXIDES_FET]
    ws, _ = chem.set_redox(w, chem.LOG_FO2_AIR, 1773.15, model="KC1991")
    ws["cr2o3"] = comp["cr2o3"]
    rows = []
    for tg_mode in ["published Tg*", "surrogate Tg*"]:
        for s in w.index:
            pub = PUB[s]
            p = sw.params(ws.loc[[s]], convention="spreadsheet",
                          tg_override=pub[1] if tg_mode == "published Tg*" else None).iloc[0]
            v = visc[visc["sample"] == s]
            cc = v[v.method == "CC"]
            if comp.loc[s, "paper"] == "AmMin":
                cc = cc[cc.T_K >= 1344 + 273.15 - 0.5]
            tg = v[v.method == "DSC"]
            rmse = np.sqrt(np.mean((sw.log_eta(p, cc.T_K) - cc.log_eta) ** 2))
            dtg = (tg.log_eta.iloc[0] - sw.log_eta(p, tg.T_K.iloc[0])) if len(tg) else np.nan
            rows.append(dict(tg_mode=tg_mode, sample=s, Cpconf=p.Cpconf, Cpconf_pub=pub[0],
                             Tg_star=p.Tg_star, Tg_star_pub=pub[1], rmse_ht=rmse, rmse_ht_pub=pub[2],
                             deta_tg=dtg, deta_tg_pub=pub[3]))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/sw2016_reproduction.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    for m, g in out.groupby("tg_mode"):
        print(f"{m}: |dRMSE| median {np.nanmedian(np.abs(g.rmse_ht - g.rmse_ht_pub)):.3f} max "
              f"{np.nanmax(np.abs(g.rmse_ht - g.rmse_ht_pub)):.3f}; |d dEta(Tg)| median "
              f"{np.nanmedian(np.abs(g.deta_tg - g.deta_tg_pub)):.2f} max "
              f"{np.nanmax(np.abs(g.deta_tg - g.deta_tg_pub)):.2f}; |dCpconf| max "
              f"{np.abs(g.Cpconf - g.Cpconf_pub).max():.2f}")


if __name__ == "__main__":
    main()
