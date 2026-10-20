"""Sehlke & Whittington (2016) GCA 191:277-299 configurational-entropy viscosity model,
with the corrected S_conf coefficients from the corrigendum (GCA 197:474-475, 2017).

    log eta = Ae + Be / (T * Sconf(T))
    Sconf(T) = Sconf(Tg*) + Cp_conf * ln(T / Tg*),   Cp_conf = Cp,l - Cp,g(Tg*)      (eq. 10)

Be and Sconf(Tg*) come from Table 6 (mol% oxides, all iron as FeO*). Be and Sconf are
therefore blind to the Fe2+/Fe3+ split; redox enters only through Cp,l.

Implementation notes
  Be, Sconf: Table 6 + corrigendum; reproduce SW Table 7 to 0.07 kJ and 0.09 J/mol/K.
  Cp,g(Tg*) = 25.19 J/mol-atom/K x atoms per mole of oxide; Cp,l = Stebbins et al. (1984);
             all iron as FeO. Reproduce SW Table 4 to within ~1.7 J/mol/K.
  Tg*: the paper uses Richet (1987) glass Cp, which we do not have. We use a ridge surrogate
       fitted to the 36 published Tg* values (leave-one-out RMSE 11 K). See TG_SURROGATE.
  Two iron conventions: 'paper' (FeO*, redox-blind) and 'spreadsheet' (what Stapperfend 2025
  ran: Fe2O3 dilutes the mol% but only FeO enters the terms).
"""
import numpy as np
import pandas as pd

AE = -3.34
B = dict(b1=1598, b2=4339, b3=275, b4=1013, b5=8, b6=15250, b7=30005)
S_CORRECTED = dict(s1=0.1615, s2=0.1419, s3=-1.0878, s4=0.0127, s5=-4.612)
S_ORIGINAL = dict(s1=0.1574, s2=0.1751, s3=-1.1056, s4=0.0123, s5=-4.958)

MW = {"sio2": 60.0843, "tio2": 79.8658, "al2o3": 101.9613, "cr2o3": 151.9904, "feo": 71.8444,
      "fe2o3": 159.6882, "mno": 70.9374, "mgo": 40.3044, "cao": 56.0774, "na2o": 61.9789,
      "k2o": 94.1960, "p2o5": 141.9445}
ATOMS = {"sio2": 3, "tio2": 3, "al2o3": 5, "cr2o3": 5, "feo": 2, "fe2o3": 5, "mno": 2, "mgo": 2,
         "cao": 2, "na2o": 3, "k2o": 3, "p2o5": 7}
# Stebbins, Carmichael & Moret (1984) liquid partial molar Cp, J/mol/K. MnO taken as FeO,
# P2O5 and Cr2O3 have no Stebbins value: taken as 3R-limit glass value (they are <1.5 mol%).
CP_LIQ = {"sio2": 80.0, "tio2": 111.8, "al2o3": 157.6, "fe2o3": 229.0, "feo": 78.9, "mno": 78.9,
          "mgo": 99.7, "cao": 99.9, "na2o": 102.3, "k2o": 97.0}
CP_ATOM = 25.19

OX = ["sio2", "tio2", "al2o3", "cr2o3", "feot", "mno", "mgo", "cao", "na2o", "k2o", "p2o5"]


def _mol_pct_feot(wt):
    """mol% over the 11 oxides with all iron as FeO*."""
    w = wt.copy()
    for c in OX:
        if c not in w:
            w[c] = 0.0
    n = pd.DataFrame({c: w[c] / (MW["feo"] if c == "feot" else MW[c]) for c in OX}, index=w.index)
    return n.div(n.sum(axis=1), axis=0) * 100


def _terms(x, fe, s):
    """Table 6 terms from mol% table x and the iron mol% `fe` that enters FeO*."""
    si, ti, al, cr, mn, mg, ca, na, k, p = (x[c] for c in
                                            ["sio2", "tio2", "al2o3", "cr2o3", "mno", "mgo", "cao", "na2o", "k2o", "p2o5"])
    A = na + k
    FM = fe + mn + mg
    exca = ((ca - (al - A)) / (fe + mg)).clip(lower=0)
    Be = (B["b1"] * (si + ti + p) + B["b2"] * al + B["b3"] * fe + B["b4"] * mg + B["b5"] * ca
          + B["b6"] * (A / ca) + B["b7"] * exca)
    SM = 0.5 * fe + mn + mg + ca + A
    nbo = 2 * (FM + ca + A - al - cr) / (si + ti + 2 * al + 2 * cr)
    Sc = (s["s1"] * (si + ti) + s["s2"] * SM + s["s3"] * (na / ca + k)
          + s["s4"] * (al + FM + ca - p) * A + s["s5"] * nbo)
    return Be, Sc, nbo


def be_sconf(wt_feot, s=S_CORRECTED):
    """Paper convention: mol% with all iron as FeO*."""
    x = _mol_pct_feot(wt_feot).rename(columns={"feot": "feo"})
    return _terms(x, x["feo"], s)


def be_sconf_spreadsheet(wt_split, s=S_CORRECTED):
    """Convention that reproduces Stapperfend 2025's SW values (Be to 0.02 %, Sconf to 0.03):
    FeO and Fe2O3 both enter the mol% normalisation, only FeO enters the formula terms."""
    w = wt_split.copy()
    cols = [c for c in MW if c != "feot"]
    for c in cols:
        if c not in w:
            w[c] = 0.0
    n = pd.DataFrame({c: w[c] / MW[c] for c in cols}, index=w.index)
    x = n.div(n.sum(axis=1), axis=0) * 100
    return _terms(x, x["feo"], s)


def heat_capacities_split(wt_split):
    """Spreadsheet convention: FeO and Fe2O3 kept separate (reproduces Stapperfend's
    Cp_conf to 0.34 J/mol/K for 15 of 16 simulants)."""
    w = wt_split.copy()
    cols = [c for c in MW if c != "feot"]
    for c in cols:
        if c not in w:
            w[c] = 0.0
    n = pd.DataFrame({c: w[c] / MW[c] for c in cols}, index=w.index)
    x = n.div(n.sum(axis=1), axis=0)
    cpg = CP_ATOM * sum(x[c] * ATOMS[c] for c in cols)
    cpl = sum(x[c] * CP_LIQ.get(c, CP_ATOM * ATOMS[c]) for c in cols)
    return cpg, cpl


def heat_capacities(wt_feot):
    """Per mole of oxides with all iron as FeO (reproduces SW Table 4 to ~0.3 J/mol/K):
    Cp,g at the 3R* limit and the Stebbins et al. (1984) liquid Cp."""
    x = _mol_pct_feot(wt_feot).rename(columns={"feot": "feo"}) / 100
    cols = list(x.columns)
    atoms = sum(x[c] * ATOMS[c] for c in cols)
    cpg = CP_ATOM * atoms
    cpl = sum(x[c] * CP_LIQ.get(c, CP_ATOM * ATOMS[c]) for c in cols)
    return cpg, cpl


# Tg* (temperature where glass Cp reaches 25.19 J/mol-atom/K, from Richet 1987 in the paper).
# Richet's coefficients are not available to us, so Tg* is a ridge surrogate on mol% (FeO*),
# fitted to the 36 published values (SW Table 4 + Stapperfend AmMin Table 5 / CG Table 7).
# Leave-one-out RMSE 11 K (max 22 K) over a 836-1085 K range.
TG_SURROGATE = dict(intercept=1019.263, sio2=2.16, tio2=-2.10, al2o3=4.06, feot=-9.88, mno=4.63,
                    mgo=-2.89, cao=-6.06, na2o=-13.83, k2o=-4.89, p2o5=19.90)


def tg_star(wt_feot):
    x = _mol_pct_feot(wt_feot)
    return TG_SURROGATE["intercept"] + sum(x[c] * v for c, v in TG_SURROGATE.items() if c != "intercept")


def params(wt_split, convention="spreadsheet", s=S_CORRECTED, tg_override=None):
    """wt_split: wt% with feo and fe2o3 (and optionally cr2o3).
    convention: 'spreadsheet' (as run by Stapperfend 2025; Fe2O3 is a diluent, so redox
    matters) or 'paper' (all iron as FeO*, redox-blind)."""
    w = wt_split.copy()
    for c in ["feo", "fe2o3", "cr2o3"]:
        if c not in w:
            w[c] = 0.0
    w["feot"] = w["feo"] + 2 * MW["feo"] / MW["fe2o3"] * w["fe2o3"]
    wf = w.drop(columns=["feo", "fe2o3"])
    if convention == "paper":
        Be, Sc, nbo = be_sconf(wf, s)
    else:
        Be, Sc, nbo = be_sconf_spreadsheet(w.drop(columns="feot"), s)
    cpg, cpl = heat_capacities(wf) if convention == "paper" else heat_capacities_split(w.drop(columns="feot"))
    tg = tg_star(wf) if tg_override is None else tg_override
    return pd.DataFrame(dict(Be=Be, Sconf=Sc, Cpg=cpg, Cpl=cpl, Cpconf=cpl - cpg, Tg_star=tg,
                             nbo_t=nbo), index=wt_split.index)


def log_eta(p_row, T):
    T = np.asarray(T, float)
    S = p_row.Sconf + p_row.Cpconf * (np.log(T) - np.log(p_row.Tg_star))
    return AE + p_row.Be / (T * S)


# ------------------------------------------------------------------ validation data (SW tables)
# Table 2 (wt%, iron as FeO* plus measured Fe2O3/FeO split where given)
SW_T2 = {
    #        SiO2  TiO2  Al2O3 Cr2O3 FeO*  MnO  MgO   CaO   Na2O K2O  P2O5  Fe2O3  FeO
    "NAK":   [46.60, 0.74, 12.11, 0.05, 20.73, 0.41, 3.97, 9.84, 2.14, 0.10, 0.28, 16.35, 6.01],
    "SHG":   [51.28, 0.75, 9.34, 0.14, 14.35, 0.15, 6.84, 12.29, 1.74, 0.18, 1.18, 10.84, 4.59],
    "EB":    [55.06, 0.18, 13.07, 0.00, 0.29, 0.14, 19.72, 12.37, 0.04, 0.08, 0.04, 0.00, 0.29],
    "EBe":   [54.80, 0.26, 15.41, 0.03, 0.26, 0.05, 12.42, 15.27, 0.04, 0.08, 0.05, 0.00, 0.26],
    "NVP":   [57.10, 0.96, 15.27, 0.67, 3.61, 0.25, 16.53, 4.95, 0.29, 0.31, 0.05, 1.89, 1.91],
    "NVPe":  [58.63, 1.28, 17.40, 0.20, 2.62, 0.07, 10.06, 6.88, 0.49, 0.45, 0.07, 1.71, 1.07],
    "NVP-Na": [54.85, 0.89, 14.84, 0.18, 2.90, 0.25, 13.58, 4.29, 6.33, 0.22, 0.00, 2.16, 0.96],
    "NVP-Nae": [59.00, 1.12, 19.25, 0.02, 2.67, 0.11, 5.81, 5.34, 7.13, 0.27, 0.01, 2.14, 0.75],
    "IcP-HCT": [53.30, 0.89, 12.31, 0.37, 3.31, 0.23, 22.53, 6.29, 0.16, 0.19, 0.04, 2.39, 1.16],
    "KREEP": [50.65, 2.35, 14.81, 0.03, 9.24, 0.17, 8.38, 10.17, 0.80, 0.79, 0.06, 6.68, 3.24],
    "KREEPe": [52.19, 2.29, 15.10, 0.01, 7.50, 0.16, 8.06, 10.69, 0.93, 0.89, 0.09, 6.17, 1.95],
    "LM":    [42.16, 9.54, 8.63, 0.12, 18.85, 0.10, 5.72, 12.68, 0.34, 0.18, 0.06, 14.67, 5.66],
    "LMe":   [49.95, 4.31, 10.94, 0.03, 9.67, 0.16, 6.03, 15.39, 0.61, 0.26, 0.16, 7.84, 2.61],
    "KOM":   [49.65, 0.36, 10.47, 0.17, 10.92, 0.11, 17.03, 9.46, 0.43, 0.14, 0.04, 7.65, 4.04],
    "EUC":   [46.96, 0.66, 12.54, 0.24, 17.52, 0.19, 7.05, 10.78, 0.46, 0.12, 0.21, 11.88, 6.83],
    "CHW":   [53.02, 1.69, 15.68, 0.02, 9.86, 0.17, 6.40, 9.04, 2.55, 0.68, 0.12, 3.40, 6.81],
    "MU":    [50.79, 2.27, 12.53, 0.01, 10.35, 0.17, 10.58, 10.50, 2.16, 0.43, 0.21, 7.82, 3.31],
    "MUe":   [54.68, 2.27, 14.45, 0.03, 6.47, 0.15, 6.78, 9.89, 2.90, 0.81, 0.45, 4.39, 2.52],
    "MU-Fe": [50.06, 2.10, 12.53, 0.01, 14.16, 0.18, 6.51, 9.17, 2.76, 0.75, 0.34, 10.55, 4.67],
    "NYI":   [39.13, 2.81, 14.90, 0.00, 12.13, 0.29, 4.08, 12.18, 5.63, 5.42, 1.48, 8.50, 4.49],
}
# Table 4: Tg3R* (K), Cp,g, Cp,l ; Table 7: Be (kJ), Sconf (original coefficients)
SW_T4 = {"NAK": (865, 69.9, 89.7), "SHG": (904, 68.6, 89.0), "EB": (1002, 69.0, 93.5),
         "EBe": (1034, 71.3, 94.1), "NVP": (1054, 71.9, 92.7), "NVPe": (1085, 74.5, 93.3),
         "NVP-Na": (972, 73.0, 94.1), "NVP-Nae": (1013, 77.2, 93.5), "IcP-HCT": (986, 68.7, 92.9),
         "KREEP": (951, 71.8, 92.8), "KREEPe": (971, 72.3, 92.9), "LM": (839, 68.0, 91.2),
         "LMe": (929, 69.9, 91.2), "KOM": (912, 67.6, 91.6), "EUC": (890, 69.5, 90.7),
         "CHW": (951, 73.1, 92.5), "MU": (915, 70.1, 92.0), "MUe": (976, 72.9, 92.2),
         "MU-Fe": (898, 70.9, 91.0), "NYI": (836, 71.4, 93.5)}
SW_T7 = {"NAK": (139.6, 9.46), "SHG": (142.3, 9.52), "EB": (148.8, 9.52), "EBe": (159.1, 10.66),
         "NVP": (157.5, 10.84), "NVPe": (164.8, 11.93), "NVP-Na": (174.4, 12.10),
         "NVP-Nae": (185.3, 13.46), "IcP-HCT": (146.0, 9.16), "KREEP": (152.6, 10.33),
         "KREEPe": (155.9, 10.69), "LM": (134.3, 8.22), "LMe": (151.9, 9.99), "KOM": (137.6, 8.05),
         "EUC": (139.1, 8.75), "CHW": (157.8, 11.38), "MU": (147.9, 10.15), "MUe": (162.2, 11.73),
         "MU-Fe": (149.0, 10.36), "NYI": (162.9, 8.75)}


def sw_table():
    cols = ["sio2", "tio2", "al2o3", "cr2o3", "feot", "mno", "mgo", "cao", "na2o", "k2o", "p2o5",
            "fe2o3", "feo"]
    return pd.DataFrame.from_dict(SW_T2, orient="index", columns=cols)


def validate():
    """Checks against the paper's own tables (paper convention)."""
    t = sw_table()
    Be, Sc, _ = be_sconf(t, S_CORRECTED)
    cpg, cpl = heat_capacities(t.drop(columns=["fe2o3", "feo"]))
    ref4 = pd.DataFrame(SW_T4, index=["Tg", "Cpg", "Cpl"]).T
    ref7 = pd.DataFrame(SW_T7, index=["Be", "Sconf"]).T
    out = pd.DataFrame(dict(dBe_kJ=Be / 1000 - ref7.Be, dSconf=Sc - ref7.Sconf, dCpg=cpg - ref4.Cpg,
                            dCpl=cpl - ref4.Cpl, dTg=tg_star(t.drop(columns=["fe2o3", "feo"])) - ref4.Tg))
    print("SW 2016 tables vs this implementation (paper convention), max |diff|:")
    print(out.abs().max().round(3).to_string())
    return out


if __name__ == "__main__":
    validate()
