"""Build the processed training table from the gpvisc `lp` sheet.

Steps
1. Load database.xlsx / lp (15,440 rows, 1 atm, wt%).
2. Keep anhydrous melts (H2O < 0.05 wt%): the lunar target is anhydrous.
3. Compute FeOT, molar Fe3+/Fe_tot, 11-oxide mole fractions, NBO/T.
4. Composition group id = cation fractions with iron as Fe_tot, rounded to 1 cation-%.
   Redox variants of the same base melt therefore share a group, so a grouped split
   never puts the oxidised and reduced versions of one melt on opposite sides.
5. Flag the controlled-fO2 redox studies (held out for the redox stress test).

Output: data/processed/train_lp.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import chem

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/raw/gpvisc/repo/code_paper_EPSL/additional_data/database.xlsx"
OUT = ROOT / "data/processed"

# Studies that vary fO2 / Fe3+ on a fixed base composition. Held out of training for the
# redox stress test. All are in gpvisc's training set, so pretrained gpvisc has seen them.
REDOX_STUDIES = [
    "Chevrel et al. 2013",     # An-Di-FeO-Fe2O3, extraterrestrial basalt analogues
    "Di Genova et al. 2017A",  # fO2 effect on Tg and viscosity, Fe-rich rhyolite
    "Dingwell 1991",           # redox viscometry of Fe-bearing melts
    "Dingwell and Virgo 1988", # Na2O-FeO-Fe2O3-SiO2
    "Casas et al. 2023",       # redox effect on molten pyrolite
    "Bouhifd et al. 2004",     # redox state of a basalt melt
    "Chevrel et al. 2014",     # Martian basalts, air + CO2-CO (redox test set)
]


def load_lp():
    d = pd.read_excel(DB, sheet_name="lp")
    d = d.rename(columns={"viscosity": "log_eta", "T": "T_K"})
    return d


def build():
    d = load_lp()
    n0 = len(d)
    d = d[d["h2o"] < 0.05].copy()
    d = d[d[chem.OXIDES].sum(axis=1) > 0]
    print(f"lp rows {n0} -> anhydrous {len(d)}")

    wt = d[chem.OXIDES].div(d[chem.OXIDES].sum(axis=1), axis=0) * 100
    d[chem.OXIDES] = wt
    d["feot"] = chem.feot_wt(d)
    d["fe3_ratio"] = chem.fe3_ratio_from_wt(d)
    x = chem.wt_to_mol(d)
    for c in chem.OXIDES:
        d["x_" + c] = x[c]
    d["nbo_t"] = chem.nbo_t(x)

    # cation fractions with total iron: invariant to the FeO/Fe2O3 split
    cat_per = {"sio2": 1, "tio2": 1, "al2o3": 2, "feot": 1, "mno": 1, "mgo": 1,
               "cao": 1, "na2o": 2, "k2o": 2, "p2o5": 2}
    n = pd.DataFrame({c: d[c] / chem.MW[c] * k for c, k in cat_per.items()}, index=d.index)
    cat = n.div(n.sum(axis=1), axis=0) * 100
    for c in cat_per:
        d["cat_" + c] = cat[c]
    key = cat.round().astype(int).astype(str).agg("-".join, axis=1)  # 1 cation-% bins
    d["group"] = pd.factorize(key)[0]
    d["is_redox_study"] = d["ref"].isin(REDOX_STUDIES)
    d["fe_rich"] = d["feot"] >= 5
    d["reduced"] = d["fe_rich"] & (d["fe3_ratio"] < 0.1)

    cols = (["Name", "ref", "group", "T_K", "log_eta"] + chem.OXIDES + ["feot", "fe3_ratio", "nbo_t"]
            + ["x_" + c for c in chem.OXIDES] + ["cat_" + c for c in chem.OXIDES_FET]
            + ["is_redox_study", "fe_rich", "reduced"])
    d = d[cols].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "train_lp.csv", index=False)

    print(f"groups {d.group.nunique()}, Fe-rich rows {d.fe_rich.sum()}, "
          f"reduced Fe-rich rows {d.reduced.sum()}, redox-study rows {d.is_redox_study.sum()}")
    for ref in REDOX_STUDIES:
        s = d[d.ref == ref]
        print(f"  {ref:26s} rows {len(s):4d} groups {s.group.nunique():3d} "
              f"Fe3+/FeT {s.fe3_ratio.min():.2f}-{s.fe3_ratio.max():.2f} "
              f"log eta {s.log_eta.min():.1f}-{s.log_eta.max():.1f}")
    return d


if __name__ == "__main__":
    build()
