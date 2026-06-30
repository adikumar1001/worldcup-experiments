"""
viz.py
------
Generates the figures referenced in the README and notebook. Each function
saves a PNG into ../figures and returns its path.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from data_loader import tournament_rates, VAR_YEAR
from module1_var import fit_match_level
from module2_shootout import run_test
from module3_format_ab import run_experiment

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(_HERE, ".", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right" : False,
    "font.size": 11,
})

def fig_var():
    """Module 1: penalty rate over time with VAR intervention marked."""
    t= tournament_rates(mens_only= True)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(t["year"], t["penalty_rate"], "o-", color="#1f77b4", label="Penalty goals / match")
    ax.axvline(VAR_YEAR, color="#d62728", ls="--", lw=2, label="VAR introduced (2018)")
    ax.set_xlabel("Tournamnet year")
    ax.set_ylabel("Penalty goals per match")
    ax.set_title("Module 1 — Penalty rate across men's World Cups")
    ax.legend()
    path = os.path.join(FIG_DIR, "module1_var.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

def fig_shootout():
    """Module 2: first-kicker win rate with CI, plus the power curve."""
    r= run_test()
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 4.5))

    rate = r["win_rate"]* 100
    lo, hi = r["ci_95"][0] * 100, r["ci_95"][1]* 100
    ax1.bar(["First kicker"], [rate], color="#2ca02c", yerr=[[rate - lo], [hi - rate]], capsize=8, width= 0.5)
    ax1.axhline(50, color="black", ls="--", label= "Chance (50%)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Win rate (%)")
    ax1.set_title(f"First-kicker win rate (n={r['n_shootouts']})\n" f"p = {r['p_value']:.3f}")
    ax1.legend()

    rates = [p[0] * 100 for p in r["power_curve"]]
    powers = [p[1] for p in r["power_curve"]]
    ax2.plot(rates, powers, "o-", color="#9467bd")
    ax2.axhline(0.8, color="#d62728", ls="--", label="80% power")
    ax2.set_xlabel("Hypothetical true win rate (%)")
    ax2.set_ylabel("Power to detect")
    ax2.set_title("Module 2 — Statistical power\n(small sample = low power)")
    ax2.legend()

    path = os.path.join(FIG_DIR, "module2_shootout.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path    


def fig_format(exp=None):
    """Module 3: distribution of champion strength under each format."""
    if exp is None:
        exp = run_experiment()
    a = exp["arm_a"]["champion_elo"]
    b = exp["arm_b"]["champion_elo"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.hist(a, bins=30, alpha=0.6, label="32-team (legacy)", color="#1f77b4")
    ax1.hist(b, bins=30, alpha=0.6, label="48-team (2026)", color="#ff7f0e")
    ax1.axvline(a.mean(), color="#1f77b4", ls="--")
    ax1.axvline(b.mean(), color="#ff7f0e", ls="--")
    ax1.set_xlabel("Champion Elo strength")
    ax1.set_ylabel("Simulated tournaments")
    ax1.set_title("Champion strength by format")
    ax1.legend()

    upa = exp["arm_a"]["knockout_upsets"]
    upb = exp["arm_b"]["knockout_upsets"]
    ax2.boxplot([upa, upb], tick_labels=["32-team", "48-team"])
    ax2.set_ylabel("Knockout upsets per tournament")
    ax2.set_title("Module 3 — Upsets by format\n(48-team is more upset-prone)")

    path = os.path.join(FIG_DIR, "module3_format.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def generate_all():
    paths = [fig_var(), fig_shootout(), fig_format()]
    print("Saved figures:")
    for p in paths:
        print("  ", p)
    return paths


if __name__ == "__main__":
    generate_all()