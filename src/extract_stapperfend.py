"""Extract the held-out lunar-simulant test set from the two Stapperfend 2025 papers.

Sources (open-access PDFs in data/raw/papers/):
  AmMin: Stapperfend et al. 2025, Am. Mineral. 110, 1171-1185  (6 simulants)
  CG:    Stapperfend et al. 2025, Chem. Geol. 696, 123082      (10 simulants)

Outputs (data/processed/):
  stapperfend_compositions.csv  wt% oxides, total iron as FeO(T); one row per simulant
  stapperfend_viscosity.csv     every measured point: concentric cylinder (CC),
                                DSC shift-factor point at Tg,DSC, and FDSC points
  stapperfend_vft.csv           the authors' own VFT fits and Tg values
  stapperfend_model_scores.csv  the authors' published RMSE / dEta(Tg) per model

Notes
- Every Fe3+/Fe_tot in both papers is *calculated* (Kress & Carmichael 1991 at air),
  not measured. We store total iron only and apply our own redox model downstream.
- Values in brackets in CG Table 1 are torque readings outside 10-90 % of full scale;
  the authors call them invalid. We keep them with valid=False.
"""
import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/papers"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

FEO_PER_FE2O3 = 2 * 71.844 / 159.688  # 0.8998: wt% Fe2O3 -> wt% FeO

# DSC shift-factor point: log eta(Tg,DSC) = K - log10(q), q = 10 K/min = 0.1667 K/s
# (both papers print the resulting log eta, so we use their printed value directly)

# ---------------------------------------------------------------- compositions
# AmMin Table 2, "This work" columns (EPMA of post-viscometry glass). Fe as Fe2O3(T).
# <d.l. entries are set to 0.
AMMIN_COMP = {
    #            SiO2   TiO2   Al2O3  Fe2O3T Na2O  K2O   MgO   CaO    MnO   Cr2O3 P2O5
    "TUBS-M":    [49.07, 2.47, 13.55, 10.93, 3.97, 1.82, 8.28, 8.25, 0.17, 0.00, 0.53],
    "TUBS-T":    [48.92, 0.17, 30.16, 1.46, 3.09, 0.29, 0.89, 13.91, 0.00, 0.00, 0.04],
    "TUBS-I":    [49.04, 1.38, 21.66, 6.44, 3.44, 1.02, 4.70, 11.22, 0.10, 0.00, 0.28],
    "Station-5": [42.75, 10.62, 13.14, 12.65, 3.11, 1.36, 7.30, 7.76, 0.28, 0.17, 0.42],
    "LRV-6":     [45.65, 5.43, 20.26, 7.70, 3.13, 0.89, 5.28, 10.37, 0.17, 0.10, 0.26],
    "LRV-7/8":   [46.55, 5.58, 15.18, 10.83, 3.41, 1.47, 6.97, 8.75, 0.21, 0.09, 0.43],
}
AMMIN_COLS = ["sio2", "tio2", "al2o3", "fe2o3t", "na2o", "k2o", "mgo", "cao", "mno", "cr2o3", "p2o5"]
AMMIN_TOTALS = {"TUBS-M": 99.04, "TUBS-T": 98.93, "TUBS-I": 99.27,
                "Station-5": 99.56, "LRV-6": 99.22, "LRV-7/8": 99.48}

# CG Table 4: 16 columns = literature composition and, where available, "Ours" (EPMA).
CG_COLS = [("LX-M100", "lit"), ("LX-T100", "lit"), ("LX-T100", "ours"), ("LMS-1", "lit"),
           ("LHS-1", "lit"), ("LHS-1", "ours"), ("JSC-1A", "lit"), ("JSC-1A", "ours"),
           ("EAC-1A", "lit"), ("OPRL2N", "lit"), ("OPRL2N", "ours"), ("OPRL2NT", "lit"),
           ("OPRH2N", "lit"), ("OPRH2N", "ours"), ("OPRH3N", "lit"), ("OPRH3N", "ours")]
CG_ROWS = {"SiO2": "sio2", "TiO2": "tio2", "Al2O3": "al2o3", "FeO": "feo", "Fe2O3": "fe2o3",
           "MnO": "mno", "MgO": "mgo", "CaO": "cao", "Na2O": "na2o", "K2O": "k2o",
           "Cr2O3": "cr2o3", "P2O5": "p2o5"}


def _num(tok):
    tok = tok.replace("−", "-").replace("− ", "-")
    if "d.l" in tok:
        return 0.0
    return float(tok)


def page_text(pdf_path, page_no):
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_no - 1].extract_text(layout=True)


def cg_compositions():
    import subprocess
    txt = subprocess.run(["pdftotext", "-f", "9", "-l", "9", "-layout",
                          str(RAW / "stapperfend_chemgeo.pdf"), "-"],
                         capture_output=True, text=True, check=True).stdout
    vals = {}
    for line in txt.splitlines():
        m = re.match(r"\s+(SiO2|TiO2|Al2O3|FeO|Fe2O3|MnO|MgO|CaO|Na2O|K2O|Cr2O3|P2O5)\s+(.*)$", line)
        if not m:
            continue
        toks = re.findall(r"<d\.l\.|\d+\.\d+", m.group(2))
        assert len(toks) == 16, (m.group(1), toks)
        vals[CG_ROWS[m.group(1)]] = [_num(t) for t in toks]
    rows = []
    for j, (name, src) in enumerate(CG_COLS):
        r = {k: v[j] for k, v in vals.items()}
        r.update(sample=name, comp_source=src, paper="CG")
        rows.append(r)
    df = pd.DataFrame(rows)
    df["feot"] = df["feo"] + df["fe2o3"] * FEO_PER_FE2O3
    # keep EPMA ("ours") when it exists, else the literature composition
    df["rank"] = (df.comp_source == "ours").astype(int)
    df = df.sort_values("rank").groupby("sample").tail(1).drop(columns=["rank", "feo", "fe2o3"])
    return df


def ammin_compositions():
    df = pd.DataFrame.from_dict(AMMIN_COMP, orient="index", columns=AMMIN_COLS)
    tot = df.sum(axis=1).round(2)
    for s, t in AMMIN_TOTALS.items():
        assert abs(tot[s] - t) < 0.02, (s, tot[s], t)
    df["feot"] = df.pop("fe2o3t") * FEO_PER_FE2O3
    df = df.reset_index().rename(columns={"index": "sample"})
    df["comp_source"] = "ours"
    df["paper"] = "AmMin"
    return df


# ---------------------------------------------------------------- viscosity
def _table_by_columns(pdf_path, page_idx, names, tcol_x, y_after, y_before):
    """Read a T-by-sample table using word x-positions so empty cells stay empty."""
    with pdfplumber.open(pdf_path) as pdf:
        ws = pdf.pages[page_idx].extract_words()
    hdr = {}
    for w in ws:
        if w["text"] in names and w["text"] not in hdr and w["top"] < y_after + 5:
            hdr[w["text"]] = (w["x0"] + w["x1"]) / 2
    centres = sorted(hdr.items(), key=lambda kv: kv[1])
    rows = {}
    for w in ws:
        if y_after < w["top"] < y_before:
            rows.setdefault(round(w["top"]), []).append(w)
    out = []
    for _, words in sorted(rows.items()):
        words.sort(key=lambda w: w["x0"])
        try:
            t_c = float(words[0]["text"])
        except ValueError:
            continue
        if not 900 < t_c < 1700:
            continue
        for w in words[1:]:
            xc = (w["x0"] + w["x1"]) / 2
            if xc < tcol_x:
                continue  # the 10000/T column
            name, cx = min(centres, key=lambda kv: abs(kv[1] - xc))
            if abs(cx - xc) > 25:
                continue
            txt = w["text"].replace("−", "-")
            valid = not (txt.startswith("(") or txt.endswith(")"))
            val = float(txt.strip("()"))
            # a minus sign printed as a separate "(cid:0)" glyph just above the number
            if any("cid:0" in v["text"] and abs(v["x0"] - w["x0"]) < 10
                   and 0 < w["top"] - v["top"] < 4 for v in ws):
                val = -val
            out.append(dict(sample=name, T_K=t_c + 273.15,
                            log_eta=val, method="CC", valid=valid))
    return pd.DataFrame(out)


def ammin_viscosity():
    names = ["TUBS-M", "TUBS-T", "TUBS-I", "Station-5", "LRV-6", "LRV-7/8"]
    with pdfplumber.open(RAW / "stapperfend_ammin.pdf") as pdf:
        ws = pdf.pages[3].extract_words()
    # this page's table has no T(°C) column in the word stream; 10000/T is at x~131
    hdr = {w["text"]: (w["x0"] + w["x1"]) / 2 for w in ws if w["text"] in names and w["top"] < 95}
    hdr.setdefault("Station-5", 368.0)  # header split across two words ("Station-5 CAS")
    hdr.setdefault("LRV-6", 435.0)
    hdr.setdefault("LRV-7/8", 501.0)
    rows = {}
    for w in ws:
        if 95 < w["top"] < 250:
            rows.setdefault(round(w["top"]), []).append(w)
    t_list = [1490, 1465, 1441, 1417, 1392, 1368, 1344, 1319, 1295, 1271, 1246,
              1222, 1198, 1173, 1149, 1125, 1100, 1076, 1051]
    out = []
    for i, (_, words) in enumerate(sorted(rows.items())):
        words.sort(key=lambda w: w["x0"])
        if float(words[0]["text"]) > 100:  # T(°C) column present in the word stream
            assert float(words[0]["text"]) == t_list[i]
            words = words[1:]
        inv_t = float(words[0]["text"])
        t_k = t_list[i] + 273.15
        assert abs(1e4 / t_k - inv_t) < 0.01, (t_list[i], inv_t)
        for w in words[1:]:
            xc = (w["x0"] + w["x1"]) / 2
            name = min(hdr, key=lambda n: abs(hdr[n] - xc))
            out.append(dict(sample=name, T_K=t_k, log_eta=float(w["text"]), method="CC", valid=True))
    return pd.DataFrame(out)


def cg_viscosity():
    names = ["LX-M100", "LX-T100", "LMS-1", "LHS-1", "EAC-1A", "JSC-1A",
             "OPRL2N", "OPRL2NT", "OPRH2N", "OPRH3N"]
    return _table_by_columns(RAW / "stapperfend_chemgeo.pdf", 4, names, tcol_x=150,
                             y_after=105, y_before=290)


# ---------------------------------------------------------------- VFT & Tg
# AmMin Table 4 and CG Table 2: A, B, C (authors' fit), Tg12 (K), TgDSC (K), log eta at TgDSC
VFT = [
    # sample,     A,     B,    C,   Tg12, TgDSC, logeta_TgDSC, paper
    ("TUBS-M",  -2.82, 3349, 711, 937, 962, 10.57, "AmMin"),
    ("TUBS-T",  -3.97, 5307, 732, 1064, 1088, 10.90, "AmMin"),
    ("TUBS-I",  -3.80, 5150, 631, 957, 987, 10.67, "AmMin"),
    ("Station-5", None, None, None, None, None, None, "AmMin"),
    ("LRV-6",   -2.96, 3458, 782, 977, 1001, 10.65, "AmMin"),
    ("LRV-7/8", -2.53, 2765, 711, 972, 993, 10.60, "AmMin"),
    ("LX-M100", None, None, None, None, None, None, "CG"),
    ("LX-T100", -3.38, 4267, 831, 836 + 273, 852 + 273, 11.14, "CG"),
    ("LMS-1",   None, None, None, None, None, None, "CG"),
    ("LHS-1",   -3.07, 3910, 788, 774 + 273, 797 + 273, 10.77, "CG"),
    ("JSC-1A",  -2.67, 2915, 737, 662 + 273, 684 + 273, 10.56, "CG"),
    ("EAC-1A",  None, None, None, None, None, None, "CG"),
    ("OPRL2N",  -2.89, 3347, 714, 666 + 273, 689 + 273, 10.59, "CG"),
    ("OPRL2NT", None, None, None, None, None, None, "CG"),
    ("OPRH2N",  -3.62, 4660, 711, 736 + 273, 761 + 273, 10.75, "CG"),
    ("OPRH3N",  None, None, None, None, None, None, "CG"),
]
# CG Table 2 prints Tg in °C (+273 applied above; the paper uses 273 not 273.15).
# AmMin Table 4 prints Tg in K. Check: VFT at Tg12 should give ~12.

# FDSC points for OPRH2N (CG Table 3): corrected peak T (°C), log eta
FDSC_OPRH2N = [(865.7, 7.09), (871.2, 6.97), (874.5, 6.87), (877.5, 6.79)]

# Published scores. AmMin Table 6 (6 models) and CG Table 6 (8 models).
CG_SCORES_TXT = """
LX-M100  0.122 - - 0.830 - - 0.192 - - 1.614 - - 0.184 - - 0.187 - - 0.498 - - 0.297 - -
LX-T100  0.507 - 6.33 0.380 - 2.46 1.248 - 4.16 0.269 - 3.40 0.550 - 1.56 0.031 - 1.30 0.207 - 2.03 0.171 - 0.93
LMS-1    0.159 - - 0.450 - - 0.109 - - 1.706 - - 0.089 - - 0.079 - - 0.450 - - 0.090 - -
LHS-1    0.438 - 5.80 0.663 - 1.40 0.602 - 2.21 0.994 - 1.08 0.164 - 1.5 0.441 - 1.65 0.105 - 0.27 0.028 - 1.23
JSC-1A   0.190 - 6.20 0.873 - -0.72 0.243 - -0.10 1.809 - -1.96 0.108 - -0.04 0.108 - 0.28 0.388 - -0.84 0.245 - -0.07
EAC-1A   0.431 - - 0.442 - - 0.243 - - 1.937 - - 0.191 - - 0.434 - - 0.517 - - 0.021 - -
OPRL2N   0.139 - 5.87 0.791 - -0.57 0.261 - -0.31 1.674 - -1.96 0.098 - -0.21 0.096 - 0.38 0.316 - -1.13 0.230 - -0.03
OPRL2NT  0.284 - - 1.001 - - 0.684 - - 1.955 - - 0.122 - - 0.618 - - 0.583 - - 0.283 - -
OPRH2N   0.502 2.948 5.76 0.752 0.726 0.78 0.890 1.085 1.64 1.151 1.015 0.16 0.199 0.194 0.57 0.515 0.204 0.90 0.152 0.938 1.05 0.081 0.530 0.26
OPRH3N   0.530 - - 0.671 - - 1.045 - - 1.012 - - 0.243 - - 0.306 - - 0.093 - - 0.115 - -
"""
CG_MODELS = ["Shaw1972", "HuiZhang2007", "GRD2008", "Duan2014", "SW2016",
             "ViscCalc2022", "GlassNet2023", "gpvisc2025"]
# NOTE on CG Table 6 OPRH2N row: printed with 23 numbers for 24 slots; gpvisc's
# entries read "0.081 0.53 0.26" so HT=0.081, FDSC=0.53, dEta(Tg)=0.26. Signs on
# JSC-1A / OPRL2N dEta(Tg) come from a (cid:0) glyph = minus sign in the PDF.


def cg_scores():
    rows = []
    for line in CG_SCORES_TXT.strip().splitlines():
        toks = line.split()
        name, vals = toks[0], toks[1:]
        assert len(vals) == 24, (name, len(vals))
        for k, m in enumerate(CG_MODELS):
            hv, fv, gv = vals[3 * k: 3 * k + 3]
            f = lambda s: None if s == "-" else float(s)
            rows.append(dict(sample=name, model=m, rmse_ht=f(hv), rmse_fdsc=f(fv),
                             deta_tg=f(gv), paper="CG"))
    return pd.DataFrame(rows)


def main():
    comp = pd.concat([ammin_compositions(), cg_compositions()], ignore_index=True)
    oxides = ["sio2", "tio2", "al2o3", "feot", "mno", "mgo", "cao", "na2o", "k2o", "p2o5", "cr2o3"]
    comp = comp[["sample", "paper", "comp_source"] + oxides]
    comp.to_csv(OUT / "stapperfend_compositions.csv", index=False)

    visc = pd.concat([ammin_viscosity(), cg_viscosity()], ignore_index=True)
    vft = pd.DataFrame(VFT, columns=["sample", "A", "B", "C", "Tg12_K", "TgDSC_K",
                                     "logeta_TgDSC", "paper"])
    dsc = vft.dropna(subset=["TgDSC_K"])
    dsc_pts = pd.DataFrame(dict(sample=dsc["sample"], T_K=dsc.TgDSC_K,
                                log_eta=dsc.logeta_TgDSC, method="DSC", valid=True))
    fdsc_pts = pd.DataFrame([dict(sample="OPRH2N", T_K=t + 273.15, log_eta=e,
                                  method="FDSC", valid=True) for t, e in FDSC_OPRH2N])
    visc = pd.concat([visc, dsc_pts, fdsc_pts], ignore_index=True)
    visc = visc.merge(comp[["sample", "paper"]], on="sample", how="left")
    visc.to_csv(OUT / "stapperfend_viscosity.csv", index=False)

    # Refit VFT to the valid CC + DSC points. AmMin Table 4 misprints C for LRV-6 and
    # LRV-7/8: the printed A,B,C miss the paper's own data by 0.7-1.1 log units, while
    # a refit reproduces the printed Tg12 within 1 K. We use the refits as reference.
    from scipy.optimize import curve_fit
    f_vft = lambda T, A, B, C: A + B / (T - C)
    refit = []
    for s_name in vft.dropna(subset=["A"])["sample"]:
        d = visc[(visc["sample"] == s_name) & visc.valid]
        p, _ = curve_fit(f_vft, d.T_K, d.log_eta, p0=[-3, 4000, 700], maxfev=20000)
        refit.append(dict(sample=s_name, A_refit=p[0], B_refit=p[1], C_refit=p[2],
                          Tg12_refit_K=p[2] + p[1] / (12 - p[0])))
    vft = vft.merge(pd.DataFrame(refit), on="sample", how="left")
    vft["printed_params_inconsistent"] = (
        (vft.A + vft.B / (vft.Tg12_K - vft.C) - 12).abs() > 0.3)
    vft.to_csv(OUT / "stapperfend_vft.csv", index=False)
    cg_scores().to_csv(OUT / "stapperfend_model_scores_cg.csv", index=False)

    # sanity checks
    ok = vft.dropna()
    tg_check = ok.A + ok.B / (ok.Tg12_K - ok.C)
    print("VFT(Tg12) should be ~12:", tg_check.round(2).tolist())
    print(comp.round(2).to_string())
    print(visc.groupby(["sample", "method"]).size().unstack(fill_value=0))
    print("invalid CC points:", (~visc.valid).sum())


if __name__ == "__main__":
    main()
