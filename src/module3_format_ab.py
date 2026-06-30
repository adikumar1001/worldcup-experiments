"""
module3_format_ab.py
--------------------
MODULE 3 — 32-team vs 48-team format: a designed A/B test (Monte Carlo)

Question
    Does the new 2026 48-team format (12 groups of 4 -> 32 advance -> Round of
    32) produce different COMPETITIVE BALANCE than the legacy 32-team format
    (8 groups of 4 -> 16 advance -> Round of 16)? Specifically: are favourites
    more or less likely to win, and do more upsets occur?

Why this is an A/B test (not just a simulation)
    We treat tournament FORMAT as the treatment with two arms:
        A (control)   = legacy 32-team structure
        B (treatment) = new 48-team structure
    We PRE-REGISTER the outcome metrics before running anything (below), use
    a common random-strength generating process so the only thing that differs
    between arms is the format, run a POWER ANALYSIS to choose the number of
    simulated tournaments, then a two-sample hypothesis test with effect sizes
    and confidence intervals. That is the anatomy of a controlled experiment;
    Monte Carlo simply supplies the randomised "users" (tournaments) because we
    cannot randomise real World Cups.

Pre-registered metrics (decided BEFORE seeing results)
    M1: champion_elo          -- strength of the eventual winner (higher = more
                                 favourite-friendly, i.e. LESS upset-prone)
    M2: top4_seed_won         -- did one of the 4 strongest entrants win? (0/1)
    M3: knockout_upsets       -- count of matches where the lower-Elo side won
    M4: champion_seed_rank    -- pre-tournament strength rank of the winner
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from elo import expected_score, strength_distribution

# --------------------------------------------------------------------------- #
# Match + knockout simulation
# --------------------------------------------------------------------------- #

def _play(r_a: float, r_b: float, rng: np.random.Generator) -> bool:
    """Return True if A beats B. Knockouts always resolve (extra time / pens)."""
    p_a= expected_score(r_a, r_b)
    return rng.random() <p_a    


def _knockout(seeds: list[int], strengths: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:

    """
    Single-elimination bracket over an even number of seeds.
    Returns (champion_seed, upset_count).
    `seeds` are indices into `strengths`. We count an upset whenever the
    lower-strength side wins a knockout match.
    """
    upsets = 0
    field = seeds[:]
    while len(field)>1:
        nxt = []
        for i in range(0, len(field),2):
            a, b =field[i], field[i+1]
            a_wins = _play(strengths[a], strengths[b], rng)
            winner, loser =(a,b) if a_wins else (b,a)
            if strengths[winner]< strengths[loser]:
                upsets +=1
            nxt.append(winner)
        field=nxt
    return field[0], upsets

# --------------------------------------------------------------------------- #
# Group stage
# --------------------------------------------------------------------------- #
def _group_stage(groups: list[list[int]], strengths: np.ndarray,
                 rng: np.random.Generator, advance_top: int,
                 best_third: int = 0):
    """
    Round-robin within each group. Top `advance_top` per group advance; plus
    the `best_third` best third-placed teams overall (used by 48-team format).
    Returns the list of advancing seed indices.
    """
    points = {s: 0 for g in groups for s in g}
    thirds = []  # (points, seed)

    for g in groups:
        # round robin
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                a, b = g[i], g[j]
                pa = expected_score(strengths[a], strengths[b])
                rv = rng.random()
                if rv < pa*0.8:  # A win
                    points[a]+=3
                elif rv >1 -(1-pa)*0.8: #b win
                    points[b]+=3
                else:
                    points[a] +=1
                    points[b] +=1
        ranked = sorted(g, key=lambda s:(points[s], strengths[s]), reverse=True)
        if len(ranked)> 2 and best_third:
            thirds.append((points[ranked[2]], strengths[ranked[2]], ranked[2]))
    advancing = []
    for g in groups:
        ranked=sorted(g, key=lambda s:(points[s], strengths[s]), reverse=True)
        advancing.extend(ranked[:advance_top])

    if best_third:
        thirds.sort(reverse=True)
        advancing.extend([t[2] for t in thirds[:best_third]])

    return advancing


# --------------------------------------------------------------------------- #
# Two formats
# --------------------------------------------------------------------------- #
def simulate_format_A(rng: np.random.Generator) -> dict:
    """Legacy 32-team: 8 groups of 4, top 2 advance (16), Round of 16 -> final."""
    strengths = strength_distribution(n=32, seed=int(rng.integers(1e9)))
    seeds = list(range(32))
    groups =[seeds[i::8] for i in range(8)]  # snake-ish split
    advancing = _group_stage(groups, strengths, rng, advance_top=2)
    advancing = sorted(advancing, key=lambda s:strengths[s], reverse= True)
    champ, upsets = _knockout(advancing, strengths, rng)
    return _metrics(champ, upsets, strengths)


def simulate_format_B(rng: np.random.Generator) -> dict:
    """New 48-team: 12 groups of 4, top2 + 8 best thirds (32), Round of 32 -> final."""
    strengths = strength_distribution(n=48, seed=int(rng.integers(1e9)))
    seeds = list(range(48))
    groups = [seeds[i::12] for i in range(12)]
    advancing = _group_stage(groups, strengths, rng, advance_top=2, best_third=8)
    advancing = sorted(advancing, key=lambda s: strengths[s], reverse=True)
    champ, upsets = _knockout(advancing, strengths, rng)
    return _metrics(champ, upsets, strengths)

def _metrics(champ: int, upsets: int, strengths: np.ndarray) -> dict:
    """Compute the four pre-registered metrics for one simulated tournament."""
    order = np.argsort(-strengths)              # strongest first
    rank_of = {seed: r for r, seed in enumerate(order)}
    return {
        "champion_elo": float(strengths[champ]),
        "top4_seed_won": int(rank_of[champ] < 4),
        "knockout_upsets": int(upsets),
        "champion_seed_rank": int(rank_of[champ]),
    }

def power_n(effect_size: float = 0.2, power: float= 0.8, alpha: float= 0.05) -> int:
     """Sims per arm to detect a small-to-medium effect (Cohen's d) at 80% power."""
     analysis = TTestIndPower()
     n = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
     return int(np.ceil(n))


def run_experiment(n_sims: int | None = None, seed: int = 42) -> dict:
    """Run both arms, then test each pre-registered metric."""
    if n_sims is None:
        n_sims=max(power_n(), 1000)
    
    rng = np.random.default_rng(seed)
    arm_a = pd.DataFrame([simulate_format_A(rng) for _ in range(n_sims)])
    arm_b = pd.DataFrame([simulate_format_B(rng) for _ in range(n_sims)])

    results = {}
    for metric in  ["champion_elo", "top4_seed_won","knockout_upsets", "champion_seed_rank"]:
        a, b = arm_a[metric].values, arm_b[metric].values
        # Welch's t-test (unequal variance) on the metric
        t, p = stats.ttest_ind(a, b, equal_var= False)
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1))/2)
        d = (b.mean() - a.mean())/ pooled_sd if pooled_sd > 0 else 0.0
        results[metric] = {
            "mean_A_32team": a.mean(),
            "mean_B_48team": b.mean(),
            "diff_B_minus_A": b.mean() - a.mean(),
            "cohens_d": d,
            "p_value": p,
            "significant": p < 0.05,
        }

    return {"n_sims_per_arm": n_sims, "arm_a": arm_a,
            "arm_b": arm_b, "results": results}

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 3 — 32-team vs 48-team Format A/B Test (Monte Carlo)")
    print("=" * 70)

    n = power_n()
    print(f"\nPower analysis: {n} sims/arm needed to detect a small effect "
          f"(d=0.2) at 80% power.\nUsing a floor of 1000/arm for stable "
          f"Monte Carlo estimates.\n")

    exp = run_experiment()
    print(f"Simulated tournaments: {exp['n_sims_per_arm']} per arm "
          f"({exp['n_sims_per_arm']*2} total)\n")

    print(f"{'Metric':<22}{'32-team':>10}{'48-team':>10}"
          f"{'Diff':>9}{'d':>7}{'p':>9}  Sig")
    print("-" * 74)
    labels = {
        "champion_elo": "Champion Elo",
        "top4_seed_won": "Top-4 seed won",
        "knockout_upsets": "Knockout upsets",
        "champion_seed_rank": "Champion rank",
    }
    for metric, r in exp["results"].items():
        print(f"{labels[metric]:<22}"
              f"{r['mean_A_32team']:>10.3f}{r['mean_B_48team']:>10.3f}"
              f"{r['diff_B_minus_A']:>9.3f}{r['cohens_d']:>7.2f}"
              f"{r['p_value']:>9.4f}  {'Yes' if r['significant'] else 'no'}")

    print("\nReading the result: a higher champion Elo / more top-4 wins under")
    print("a format means it is MORE favourite-friendly (fewer upsets). The")
    print("48-team format adds a Round of 32, giving strong teams an extra")
    print("single-elimination hurdle — the test quantifies whether that")
    print("measurably changes who lifts the trophy.")