"""Controlled-fO2 redox test set, with Fe2+/Fe_tot *measured* by titration.

Source: M.O. Chevrel (2013) PhD thesis, LMU Munich (open access, edoc.ub.uni-muenchen.de/15783),
  ch. 2 = Chevrel et al. 2013 GCA (An42Di58 + Fe, "AD" series),
  ch. 4 = Chevrel et al. 2014 GCA (Martian basalts).

Two regimes:
  HT: isothermal concentric-cylinder viscosity at stepwise CO2-CO fO2 (Tables 2.4, 4.4)
  Tg: limiting fictive temperature Tf' at 10 K/min vs redox (Tables 2.5, 4.5)

Quality flags
  - Chevrel notes Fe2+/Fe_tot from dip-quenched reduced melts may read too oxidised.
  - Ch. 4 Tf' of the most oxidised Martian glasses is biased by crystallisation above Tg
    (thesis p. 98); those rows get quality="low". BAMB-S0 is at a different T (1495 °C).
  - YMB has no titration data and is omitted.

Both source papers are in gpvisc's training set (as "Chevrel et al. 2013/2014"), so these
studies are excluded from our training data for the redox stress test.

Output: data/processed/redox_series.csv
"""
from pathlib import Path

import pandas as pd

import chem

ROOT = Path(__file__).resolve().parents[1]

# wt% (AD: Table 2.1, iron reported as Fe2O3 total; Mars: Table 4.1, FeO total)
COMP = {
    "AD-5":  dict(sio2=47.39, al2o3=14.22, feot=5.06 * chem.FEO_PER_FE2O3, mgo=9.76, cao=22.41),
    "AD-10": dict(sio2=45.64, al2o3=13.52, feot=9.03 * chem.FEO_PER_FE2O3, mgo=9.24, cao=21.18),
    "AD-20": dict(sio2=40.42, al2o3=11.93, feot=20.20 * chem.FEO_PER_FE2O3, mgo=8.12, cao=18.49),
    "AdMB": dict(sio2=47.01, tio2=0.59, al2o3=10.86, feot=19.43, mno=0.42, mgo=10.14, cao=8.09,
                 na2o=2.66, k2o=0.15, p2o5=0.65),
    "IAMB": dict(sio2=49.09, tio2=1.10, al2o3=8.58, feot=20.36, mno=0.38, mgo=9.82, cao=5.99,
                 na2o=3.10, k2o=0.62, p2o5=0.97),
    "BAMB": dict(sio2=50.22, tio2=0.94, al2o3=13.46, feot=13.87, mno=0.24, mgo=8.43, cao=6.12,
                 na2o=4.21, k2o=1.09, p2o5=1.41),
    "OMB":  dict(sio2=46.72, tio2=0.56, al2o3=10.61, feot=18.99, mno=0.11, mgo=11.39, cao=8.41,
                 na2o=2.25, k2o=0.17, p2o5=0.79),
}

# (series, sample, T_C, Fe2+/Fe_tot, log eta)  — HT regime
HT = [
    # Table 2.4: AD-10 at 1623 K, AD-20 at 1658 K (the paper gives K)
    ("AD-10", "#22", 1623 - 273.15, 0.18, 0.56), ("AD-10", "#24", 1623 - 273.15, 0.32, 0.54),
    ("AD-10", "#26", 1623 - 273.15, 0.50, 0.53), ("AD-10", "#28", 1623 - 273.15, 0.73, 0.53),
    ("AD-10", "#31", 1623 - 273.15, 0.74, 0.54), ("AD-10", "#32", 1623 - 273.15, 0.17, 0.56),
    ("AD-20", "#S1", 1658 - 273.15, 0.18, -0.121), ("AD-20", "#S2", 1658 - 273.15, 0.36, -0.198),
    ("AD-20", "#S3", 1658 - 273.15, 0.52, -0.232), ("AD-20", "#S4", 1658 - 273.15, 0.67, -0.236),
    # Table 4.4 (°C). FeO/FeOtot = Fe2+/Fe_tot.
    ("AdMB", "S2", 1385, 0.21, 0.21), ("AdMB", "S3", 1385, 0.33, 0.15), ("AdMB", "S4", 1385, 0.37, 0.14),
    ("AdMB", "S5", 1385, 0.48, 0.12), ("AdMB", "S6", 1385, 0.63, 0.09), ("AdMB", "S8", 1385, 0.78, 0.17),
    ("BAMB", "S1", 1395, 0.62, 0.56), ("BAMB", "S2", 1395, 0.64, 0.56),
    ("BAMB", "S3", 1395, 0.84, 0.55), ("BAMB", "S4", 1395, 0.77, 0.58),
    ("IAMB", "S1", 1395, 0.21, 0.32), ("IAMB", "S2", 1395, 0.40, 0.22), ("IAMB", "S3", 1395, 0.69, 0.16),
    ("IAMB", "S4", 1395, 0.82, 0.14), ("IAMB", "S5", 1395, 0.88, 0.17),
    ("OMB", "S1", 1395, 0.23, -0.04), ("OMB", "S2", 1395, 0.50, -0.09),
    ("OMB", "S3", 1395, 0.50, -0.13), ("OMB", "S4", 1395, 0.80, -0.09),
]

# (series, sample, Fe2+/Fe_tot, Tf' at 10 K/min in K, quality) — Tg regime
TG = [
    # Table 2.5 (K)
    ("AD-5", "#42", 0.17, 985, "high"), ("AD-5", "#41", 0.32, 983, "high"),
    ("AD-5", "#40", 0.70, 983, "high"), ("AD-5", "#39", 0.71, 989, "high"),
    ("AD-10", "#22", 0.18, 967, "high"), ("AD-10", "#24", 0.32, 963, "high"),
    ("AD-10", "#26", 0.50, 958, "high"), ("AD-10", "#31", 0.74, 961, "high"),
    ("AD-10", "#28", 0.73, 962, "high"),
    ("AD-20", "#S1", 0.18, 950, "medium"),  # Tf' from Tg-onset correlation (crystallises)
    ("AD-20", "#S2", 0.36, 930, "high"), ("AD-20", "#S3", 0.52, 924, "high"),
    ("AD-20", "#S4", 0.67, 924, "high"),
    # Table 4.5 (°C), only samples that also have titration data in Table 4.4
    ("AdMB", "S4", 0.37, 664 + 273.15, "low"), ("AdMB", "S5", 0.48, 638 + 273.15, "medium"),
    ("AdMB", "S6", 0.63, 620 + 273.15, "medium"), ("AdMB", "S8", 0.78, 628 + 273.15, "medium"),
    ("BAMB", "S1", 0.62, 653 + 273.15, "medium"), ("BAMB", "S2", 0.64, 629 + 273.15, "medium"),
    ("BAMB", "S3", 0.84, 628 + 273.15, "medium"), ("BAMB", "S4", 0.77, 641 + 273.15, "medium"),
    ("IAMB", "S3", 0.69, 627 + 273.15, "medium"), ("IAMB", "S4", 0.82, 619 + 273.15, "medium"),
    ("IAMB", "S5", 0.88, 615 + 273.15, "medium"),
    ("OMB", "S2", 0.50, 629 + 273.15, "medium"), ("OMB", "S3", 0.50, 632 + 273.15, "medium"),
    ("OMB", "S4", 0.80, 624 + 273.15, "medium"),
]


def build():
    rows = []
    for s, smp, t_c, fe2, eta in HT:
        rows.append(dict(series=s, sample=smp, regime="HT", T_K=t_c + 273.15, fe3_ratio=1 - fe2,
                         log_eta=eta, Tf_K=None, quality="high"))
    for s, smp, fe2, tf, q in TG:
        rows.append(dict(series=s, sample=smp, regime="Tg", T_K=None, fe3_ratio=1 - fe2,
                         log_eta=None, Tf_K=tf, quality=q))
    df = pd.DataFrame(rows)
    comp = pd.DataFrame.from_dict(COMP, orient="index").fillna(0.0)
    for c in chem.OXIDES_FET:
        if c not in comp:
            comp[c] = 0.0
    comp = comp[chem.OXIDES_FET]
    comp = comp.div(comp.sum(axis=1), axis=0) * 100
    df = df.join(comp, on="series")
    out = ROOT / "data/processed/redox_series.csv"
    df.to_csv(out, index=False)
    print(df.groupby(["series", "regime"]).agg(n=("sample", "size"),
          fe3_min=("fe3_ratio", "min"), fe3_max=("fe3_ratio", "max"),
          feot=("feot", "first")).round(2))
    return df


if __name__ == "__main__":
    build()
