"""Baseline viscosity models.

GRD 2008  Giordano, Russell & Dingwell (2008) EPSL 271, 123-134. VFT with A = -4.55.
          Iron enters only as FeO(T): the model is blind to redox by construction.
          (gpvisc's utils.G2008 drops Fe2O3 entirely and omits MnO from the b3 term,
          so we implement it here from the paper's Table 1.)
gpvisc    Le Losq et al. (2025), pretrained GP model 1, version 0.3.5 (as in Stapperfend 2025).
SW2016    Sehlke & Whittington (2016) GCA 191 - NOT implemented: coefficients are in the
          paywalled paper. Scores on the simulants are taken from Stapperfend's tables.
"""
import numpy as np
import pandas as pd

import chem

GRD_B = dict(b1=159.56, b2=-173.34, b3=72.13, b4=75.69, b5=-38.98, b6=-84.08, b7=141.54,
             b11=-2.43, b12=-0.91, b13=17.62)
GRD_C = dict(c1=2.75, c2=15.72, c3=8.32, c4=10.2, c5=-12.29, c6=-99.54, c11=0.3)
GRD_A = -4.55


def grd2008_params(df_wt_feot):
    """VFT A, B, C from wt% with total iron as 'feot'. Anhydrous (V = H2O + F2O-1 = 0)."""
    x = chem.wt_to_mol(df_wt_feot, chem.OXIDES_FET) * 100  # mol%
    si, ti, al, fe, mn, mg, ca, na, k, p = (x[c] for c in chem.OXIDES_FET)
    h2o = 0.0
    V = 0.0
    TA = ti + al
    FM = fe + mn + mg
    NK = na + k
    B = (GRD_B["b1"] * (si + ti) + GRD_B["b2"] * al + GRD_B["b3"] * (fe + mn + p)
         + GRD_B["b4"] * mg + GRD_B["b5"] * ca + GRD_B["b6"] * (na + V)
         + GRD_B["b7"] * (V + np.log1p(h2o)) + GRD_B["b11"] * (si + ti) * FM
         + GRD_B["b12"] * (si + TA + p) * (NK + h2o) + GRD_B["b13"] * al * NK)
    C = (GRD_C["c1"] * si + GRD_C["c2"] * TA + GRD_C["c3"] * FM + GRD_C["c4"] * ca
         + GRD_C["c5"] * NK + GRD_C["c6"] * np.log1p(V) + GRD_C["c11"] * (al + FM + ca - p) * (NK + V))
    return pd.DataFrame({"A": GRD_A, "B": B, "C": C}, index=df_wt_feot.index)


def vft(T, A, B, C):
    return A + B / (np.asarray(T) - C)


def vft_T_at(log_eta, A, B, C):
    """Temperature (K) at which the VFT curve reaches log_eta."""
    return C + B / (log_eta - A)


# ------------------------------------------------------------------ gpvisc
_GP = None


def _gp():
    global _GP
    if _GP is None:
        import warnings
        warnings.filterwarnings("ignore")
        import gpvisc
        _GP = gpvisc.load_gp_model(model_number=1, device="cpu")
    return _GP


def gpvisc_predict(df_wt, T_K, return_std=True):
    """df_wt: wt% with feo and fe2o3 columns (redox already set). Returns mean, std (log Pa s)."""
    import gpvisc
    gp_model, likelihood = _gp()
    q = df_wt.copy()
    for c in gpvisc.list_oxides():
        if c not in q:
            q[c] = 0.0
    q = q[gpvisc.list_oxides()].reset_index(drop=True)
    q = gpvisc.chimie_control(q)
    q = gpvisc.wt_mol(q)
    T = np.asarray(T_K, dtype=float)
    P = np.zeros_like(T)
    X = gpvisc.scale_for_gaussianprocess(pd.Series(T), pd.Series(P), q)
    mean, std = gpvisc.predict(X, gp_model, likelihood)
    mean, std = np.asarray(mean).ravel(), np.asarray(std).ravel()
    return (mean, std) if return_std else mean
