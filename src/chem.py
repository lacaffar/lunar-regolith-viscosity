"""Melt chemistry: wt% -> mol fractions, total iron, iron redox, oxygen buffers, NBO/T.

Conventions used everywhere in this project
- Compositions carry 11 anhydrous oxides. Cr2O3 (<=0.2 wt% in our samples) is dropped
  before normalising. H2O is excluded: lunar melts are anhydrous and so is our training set.
- Iron enters as FeO + Fe2O3. Total iron as FeO: FeOT = FeO + 0.8998 * Fe2O3 (wt%),
  i.e. in moles n(Fe_tot) = n(FeO) + 2 * n(Fe2O3).
- Redox = Fe3+/Fe_tot (molar). For a target fO2 it is set with Borisov et al. (2018),
  using the implementation shipped in gpvisc 0.3.5 (the version Stapperfend 2025 used).
"""
import numpy as np
import pandas as pd

OXIDES = ["sio2", "tio2", "al2o3", "feo", "fe2o3", "mno", "mgo", "cao", "na2o", "k2o", "p2o5"]
OXIDES_FET = ["sio2", "tio2", "al2o3", "feot", "mno", "mgo", "cao", "na2o", "k2o", "p2o5"]

MW = {  # g/mol, IUPAC 2016 atomic weights (same as gpvisc)
    "sio2": 60.0843, "tio2": 79.8658, "al2o3": 101.9613, "feo": 71.8444, "fe2o3": 159.6882,
    "mno": 70.9374, "mgo": 40.3044, "cao": 56.0774, "na2o": 61.9789, "k2o": 94.1960,
    "p2o5": 141.9445, "feot": 71.8444,
}
FEO_PER_FE2O3 = 2 * MW["feo"] / MW["fe2o3"]  # 0.8998


def wt_to_mol(df, cols=OXIDES):
    """wt% oxides -> mole fractions summing to 1 over `cols`."""
    n = pd.DataFrame({c: df[c] / MW[c] for c in cols}, index=df.index)
    return n.div(n.sum(axis=1), axis=0)


def mol_to_wt(df, cols=OXIDES):
    w = pd.DataFrame({c: df[c] * MW[c] for c in cols}, index=df.index)
    return w.div(w.sum(axis=1), axis=0) * 100


def feot_wt(df):
    return df["feo"] + FEO_PER_FE2O3 * df["fe2o3"]


def fe3_ratio_from_wt(df):
    """Molar Fe3+/Fe_tot from wt% FeO and Fe2O3."""
    n2 = df["feo"] / MW["feo"]
    n3 = 2 * df["fe2o3"] / MW["fe2o3"]
    tot = n2 + n3
    return (n3 / tot.replace(0, np.nan)).fillna(0.0)


def split_iron(df_wt, fe3_ratio):
    """Given wt% with total iron in column 'feot', return wt% with feo/fe2o3 split
    so that molar Fe3+/Fe_tot = fe3_ratio. Other oxides untouched (sum changes slightly)."""
    out = df_wt.copy()
    n_fe = out["feot"] / MW["feo"]
    out["feo"] = n_fe * (1 - fe3_ratio) * MW["feo"]
    out["fe2o3"] = n_fe * fe3_ratio / 2 * MW["fe2o3"]
    return out.drop(columns="feot")


# ----------------------------------------------------------------- oxygen buffers
def log_fo2_iw(T_K):
    """Iron-wustite buffer at 1 bar, log10 fO2 (bar). O'Neill & Pownceby (1993)-type
    fit as used widely in planetary petrology: log fO2 = 6.57 - 27215/T (Huebner 1971).
    The two common IW parameterisations differ by <0.1 log units at 1400-1800 K."""
    return 6.57 - 27215.0 / np.asarray(T_K, dtype=float)


LOG_FO2_AIR = np.log10(0.2095)


# ----------------------------------------------------------------- redox models
def _mol_pct_for_redox(df_wt_feot):
    """Mole fractions over the 10 oxides with total iron as FeO (input form used by
    the 1-atm redox parameterisations' composition terms)."""
    return wt_to_mol(df_wt_feot, OXIDES_FET).rename(columns={"feot": "feo"})


def fe3_borisov2018(df_wt_feot, log_fo2, T_K):
    """Borisov, Behrens & Holtz (2018) Fe3+/Fe_tot. Coefficients and form copied from
    gpvisc.redox.redox_B2018 (Le Losq et al.). Composition terms use mole fractions."""
    x = _mol_pct_for_redox(df_wt_feot)
    xdi = (x["sio2"] * -0.445 + x["tio2"] * -0.900 + x["mgo"] * 1.532 + x["cao"] * 0.314
           + x["na2o"] * 2.030 + x["k2o"] * 3.355 + x["p2o5"] * -4.851
           + x["sio2"] * x["al2o3"] * -3.081 + x["sio2"] * x["mgo"] * -4.370)
    log_r = 0.207 * np.asarray(log_fo2) + 4633.3 / np.asarray(T_K) - 1.852 + xdi.values
    r = 10 ** log_r  # Fe3+/Fe2+
    return r / (1 + r)


def fe3_kc1991(df_wt_feot, log_fo2, T_K):
    """Kress & Carmichael (1991) 1-atm Fe3+/Fe_tot (the model Stapperfend 2025 used)."""
    x = _mol_pct_for_redox(df_wt_feot)
    xdi = (x["al2o3"] * -2.243 + x["feo"] * -1.828 + x["cao"] * 3.201
           + x["na2o"] * 5.854 + x["k2o"] * 6.215)
    ln_r = 0.196 * np.log(10 ** np.asarray(log_fo2)) + 1.1492e4 / np.asarray(T_K) - 6.675 + xdi.values
    r = 2 * np.exp(ln_r)  # XFe2O3/XFeO -> Fe3+/Fe2+
    return r / (1 + r)


REDOX_MODELS = {"B2018": fe3_borisov2018, "KC1991": fe3_kc1991}


def set_redox(df_wt_feot, log_fo2, T_K, model="B2018"):
    """Return (wt% table with FeO/Fe2O3 split, Fe3+/Fe_tot array)."""
    r = REDOX_MODELS[model](df_wt_feot, log_fo2, T_K)
    return split_iron(df_wt_feot, r), r


# ----------------------------------------------------------------- structure
def nbo_t(x_mol):
    """NBO/T from mole fractions (Mysen 1988 style). Fe3+ treated as network former
    (charge-balanced, like Al), Fe2+ as modifier. Ti as former."""
    si, ti, al = x_mol["sio2"], x_mol["tio2"], 2 * x_mol["al2o3"]
    fe3 = 2 * x_mol["fe2o3"]
    T = si + ti + al + fe3
    O = (2 * x_mol["sio2"] + 2 * x_mol["tio2"] + 3 * x_mol["al2o3"] + 3 * x_mol["fe2o3"]
         + x_mol["feo"] + x_mol["mno"] + x_mol["mgo"] + x_mol["cao"] + x_mol["na2o"]
         + x_mol["k2o"] + 5 * x_mol["p2o5"])
    # NBO = 2*O - 4*T (each tetrahedral cation shares 4 oxygens)
    return ((2 * O - 4 * T) / T).clip(lower=0)


def sm_giordano(x_mol):
    """Structure-modifier content SM (mol%) of Giordano & Dingwell 2003."""
    return 100 * (x_mol["feo"] + x_mol["mno"] + x_mol["mgo"] + x_mol["cao"]
                  + x_mol["na2o"] + x_mol["k2o"])
