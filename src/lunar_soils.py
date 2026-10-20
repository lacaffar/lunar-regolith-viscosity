"""Application set: bulk chemistry of returned lunar soils.

Source: Zenodo record 5899696, Table S2 "Bulk chemical data of lunar soils" (compiled from
Apollo, Luna and Chang'E-5 literature; references listed in the sheet). FeO is total iron.

Kept: soils with all major oxides reported (SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O).
Chang'E-5: the three replicate analyses are replaced by the reported soil average.
Cr2O3 is dropped (<0.5 wt%); missing MnO/K2O/P2O5 are set to 0.

Output: data/processed/lunar_soils.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import chem

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/zenodo_5899696/sm/Table s2 Bulk chemical data of lunar soils.xlsx"
COLS = ["sio2", "tio2", "al2o3", "feot", "mno", "mgo", "cao", "na2o", "k2o", "cr2o3", "p2o5"]
MAJORS = ["sio2", "tio2", "al2o3", "feot", "mgo", "cao", "na2o"]


def mission(sample):
    s = str(sample).strip()
    if s.startswith("Luna"):
        return s
    if s.startswith("CE"):
        return "Chang'e-5"
    try:
        n = int(float(s))
    except ValueError:
        return "other"
    first = str(n)[:2] if n >= 10000 else str(n)[:1]
    return {"10": "Apollo 11", "12": "Apollo 12", "14": "Apollo 14", "15": "Apollo 15",
            "6": "Apollo 16", "7": "Apollo 17"}.get(first if n < 60000 else str(n)[0], "other")


def build():
    raw = pd.read_excel(SRC, header=None)
    rows = []
    for i in range(1, 127):
        r = raw.iloc[i]
        name = str(r[3]).strip()
        if name in ("nan", "CE5C0800YJFM002") or name.startswith("CE-5 basalt"):
            continue
        vals = pd.to_numeric(r[5:16], errors="coerce").to_numpy(float)
        rows.append(dict(sample=name.replace("CE-5 soil average", "CE-5 soil"),
                         ref=str(r[4]).replace("\n", "; "), **dict(zip(COLS, vals))))
    d = pd.DataFrame(rows)
    d["mission"] = d["sample"].map(mission)
    n_all = len(d)
    d = d.dropna(subset=MAJORS).copy()
    for c in ["mno", "k2o", "p2o5"]:
        d[c] = d[c].fillna(0.0)
    d = d.drop(columns="cr2o3")
    ox = chem.OXIDES_FET
    d["sum_raw"] = d[ox].sum(axis=1)
    d[ox] = d[ox].div(d[ox].sum(axis=1), axis=0) * 100
    # simple terrane label: highland soils are Al-rich and Fe-poor
    d["terrane"] = np.where(d.al2o3 >= 20, "highland", np.where(d.feot >= 14, "mare", "mixed"))
    d = d.drop_duplicates(subset=["sample"]).reset_index(drop=True)
    d.to_csv(ROOT / "data/processed/lunar_soils.csv", index=False)
    print(f"{n_all} soil rows -> {len(d)} with full majors")
    print(d.groupby("mission").agg(n=("sample", "size"), FeO_min=("feot", "min"),
                                   FeO_max=("feot", "max")).round(1))
    print(d.terrane.value_counts())
    return d


if __name__ == "__main__":
    build()
