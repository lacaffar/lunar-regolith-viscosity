"""Figures for RESULTS.md (static PNG, light theme, validated 3-slot palette + marker shapes)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
F = ROOT / "results/figures"
F.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def fig_redox():
    s = pd.read_csv(T / "redox_seriesA_slopes.csv")
    anc = pd.read_csv(T / "redox_anchor_series.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    models = [("gpvisc 0.3.5", S1, "s"), ("Deep ensemble", S2, "o"), ("Sehlke & Whittington 2016", S3, "^")]
    for ax, regime, unit, scale in [(axes[0], "HT", "Δlog η (Pa s) on the Moon", 1),
                                    (axes[1], "Tg", "ΔTg on the Moon (K)", 1)]:
        g = s[s.regime == regime]
        order = g.groupby("series").feot.first().sort_values()
        x = np.arange(len(order))
        dfe3 = -0.58  # air (1500 °C) -> IW-1
        a = anc[anc.regime == regime].set_index("series").loc[order.index]
        ax.axhline(0, color=NEUTRAL, lw=1.5, ls="--", zorder=1)
        ax.text(0.01, 0.01, "dashed line = GRD 2008 (redox-blind, always 0)", color=INK2,
                fontsize=8, ha="left", va="bottom", transform=ax.transAxes)
        ax.errorbar(x, a.slope * dfe3, yerr=1.96 * a.se * abs(dfe3), fmt="D", ms=7, color=INK,
                    ecolor=INK, elinewidth=1.5, capsize=3, label="Measured (titrated Fe²⁺), 95% CI",
                    zorder=4)
        for k, (m, c, mk) in enumerate(models):
            v = g[g.model == m].set_index("series").loc[order.index].predicted_slope * dfe3
            ax.plot(x + (k - 1) * 0.18, v, mk, ms=8, color=c, mec=SURF, mew=1.5, zorder=3,
                    label=m + (" (as run by Stapperfend)" if m.startswith("Sehlke") else ""))
        ax.set_xticks(x, [f"{n}\n{f:.0f}% FeO" for n, f in order.items()], fontsize=8.5)
        ax.set_ylabel(unit)
        ax.set_title({"HT": "Fully molten (~1400 °C): measured effect is tiny",
                      "Tg": "Near Tg: measured effect is large, models disagree"}[regime],
                     loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Redox stress test: Earth-test (air) → lunar (IW−1) shift, measured vs predicted",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F / "redox_stress_test.png", dpi=170)
    plt.close(fig)


def fig_earth_vs_moon():
    d = pd.read_csv(T / "earth_vs_moon_all.csv")
    levels = ["working (10^3)", "softening (10^6.6)", "Tg (10^12)"]
    titles = {"working (10^3)": "Working point, 10³ Pa·s", "softening (10^6.6)": "Softening, 10⁶·⁶ Pa·s",
              "Tg (10^12)": "Glass transition, 10¹² Pa·s"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, lvl in zip(axes, levels):
        g = d[d.level == lvl].sort_values("FeOT")
        ax.axhline(0, color=NEUTRAL, lw=1, zorder=1)
        ax.errorbar(g.FeOT - 0.12, g.dT_ensemble, yerr=[g.dT_ensemble - g.dT_ens_lo, g.dT_ens_hi - g.dT_ensemble],
                    fmt="o", ms=5, color=S2, ecolor=S2, elinewidth=1, alpha=0.85, mec=SURF, mew=1,
                    label="Deep ensemble (member spread, 95%)", zorder=3)
        lo = np.minimum(g.dT_anc_lo, g.dT_sat_lo)
        hi = np.maximum(g.dT_anc_hi, g.dT_sat_hi)
        ax.vlines(g.FeOT + 0.12, lo, hi, color=S1, lw=2.2, alpha=0.55, zorder=3,
                  label="Measurement-anchored bracket (95%)")
        ax.plot(g.FeOT + 0.12, g.dT_anchored, "D", ms=4.5, color=S1, mec=SURF, mew=0.8, zorder=4,
                label="  linear in Fe³⁺/Feₜₒₜ (upper bound)")
        ax.plot(g.FeOT + 0.12, g.dT_anchored_sat, "D", ms=4.5, mfc=SURF, mec=S1, mew=1.2, zorder=4,
                label="  saturating below Fe³⁺/Feₜₒₜ = 0.5 (lower bound)")
        ax.set_title(titles[lvl], loc="left")
        ax.set_xlabel("Soil FeO (total Fe, wt%)")
    axes[0].set_ylabel("T(Moon, IW−1) − T(Earth test, air)  [K]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Processing-temperature shift, Earth test (air, 1500 °C melt) → Moon (IW−1), 66 returned soils",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F / "earth_vs_moon_shift.png", dpi=170)
    plt.close(fig)


def fig_scorecard():
    p = pd.read_csv(T / "scorecard_per_sample.csv")
    p = p[p.deta_tg.notna()]
    models = [("gpvisc 0.3.5", S1, "s"), ("Sehlke & Whittington 2016 (published)", S3, "^"),
              ("Deep ensemble", S2, "o")]
    order = p[p.model == "gpvisc 0.3.5"].set_index("sample").deta_tg.sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhspan(-0.5, 0.5, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax.axhline(0, color=NEUTRAL, lw=1)
    x = np.arange(len(order))
    for k, (m, c, mk) in enumerate(models):
        v = p[p.model == m].set_index("sample").reindex(order).deta_tg
        ax.plot(x + (k - 1) * 0.2, v, mk, ms=8, color=c, mec=SURF, mew=1.5, label=m.replace(" (published)", ""))
    g = p[p.model == "GRD 2008"].set_index("sample").reindex(order).deta_tg
    ax.plot(x, g, "_", ms=12, color=NEUTRAL, mew=2, label="GRD 2008")
    ax.set_xticks(x, order, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Δη(Tg) = measured − model (log units)")
    ax.text(-0.4, -0.5, "±0.5 band", color=INK2, fontsize=8, ha="left", va="top")
    ax.set_title("Held-out simulants: error at the glass transition (10 with DSC data)", loc="left")
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(F / "scorecard_tg.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    for f in [fig_redox, fig_scorecard, fig_earth_vs_moon]:
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
