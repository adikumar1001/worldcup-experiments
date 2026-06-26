"""
module2_shootout.py
-------------------
MODULE 2 — Does kicking first win shootouts? (Natural Experiment)

Question
    In a penalty shootout, does the team that kicks FIRST win more often than
    chance (50%)?

Why this is a natural experiment
    Which team kicks first is decided by a coin toss. Coin tosses are random,
    so "kicks first" is effectively RANDOMLY ASSIGNED across shootouts. That
    randomisation is what lets us interpret any win-rate gap causally, rather
    than as mere correlation — without us having to run an experiment
    ourselves. The world already ran it.

Method
    - Reconstruct, for each shootout, which team kicked first (first kick in
      the ordered kick log) and which team ultimately won.
    - Binomial test of first-kicker win rate against the 50% null.
    - Report the 95% confidence interval (Wilson) and, crucially, a POWER
      analysis: with only ~40 shootouts, how large an advantage could this
      sample realistically detect? Honest treatment of power is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from data_loader import load_penalty_kicks, load_matches

def build_shootout_table() -> pd.DataFrame: 
        """One row per shootout: first-kicking team and whether they won."""
        pk= load_penalty_kicks()
        matches= load_matches()   
    # First kicker = team of the first kick in each shootout (ordered by key_id).
        first_kicker = ( pk.groupby("match_id").first()["team_name"].rename("first_kick_team"))

        ms= matches[matches["penalty_shootout"] == 1].copy()
        ms= ms.merge(first_kicker, on="match_id", how="inner")

    # Determine the shootout winner from the penalty score.
        def winner(row):
            if row["home_team_score_penalties"]> row["away_team_score_penalties"]:
                return row["home_team_name"]
            if row["away_team_score_penalties"]> row["home_team_score_penalties"]:
                return row["away_team_name"]
            return None # should not happen in a decided shootout

        ms["shootout_winner"]= ms.apply(winner, axis=1)
        ms= ms.dropna(subset=["shootout_winner"])
        ms["first_kicker_won"]= (ms["first_kick_team"]== ms["shootout_winner"]).astype(int)
        return ms[["match_name", "match_date", "first_kick_team", "shootout_winner", "first_kicker_won"]]


def run_test() -> dict:
      """Binomial test + CI + power analysis on first-kicker win rate."""
      tbl =build_shootout_table()
      n= len(tbl)
      wins=int (tbl["first_kicker_won"].sum())
      rate = wins/n
      
       # Two-sided exact binomial test against p = 0.5
      binom = stats.binomtest(wins, n, 0.5, alternative="two-sided")
      ci_low, ci_high= proportion_confint(wins, n ,alpha=0.05, method="wilson")

      
        # Power: what effect could a sample of this size detect at 80% power?
      analysis = NormalIndPower()
      detectable = []
      for true_rate in np.arange(0.55, 0.86, 0.05):
          es=proportion_effectsize(true_rate, 0.5)
          pwr = analysis.power(effect_size=es, nobs1=n, alpha=0.05, ratio=0)
          detectable.append((round(true_rate,2), round(pwr, 2)))


      return {
        "n_shootouts": n,
        "first_kicker_wins": wins,
        "win_rate": rate,
        "p_value": binom.pvalue,
        "ci_95": (ci_low, ci_high),
        "power_curve": detectable,
        "table": tbl,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 2 — Shootout First-Kicker Advantage (Natural Experiment)")
    print("=" * 70)

    r = run_test()
    print(f"\nShootouts analysed       : {r['n_shootouts']}")
    print(f"First kicker won         : {r['first_kicker_wins']} "
          f"({r['win_rate']*100:.1f}%)")
    print(f"95% CI (Wilson)          : "
          f"[{r['ci_95'][0]*100:.1f}%, {r['ci_95'][1]*100:.1f}%]")
    print(f"Binomial test vs 50%     : p = {r['p_value']:.4f}")
    verdict = ("significant first-kicker advantage"
               if r["p_value"] < 0.05 else
               "no significant advantage detected")
    print(f"Verdict                  : {verdict}")

    print("\nPower analysis (can this sample even detect an advantage?):")
    print("  If the TRUE win rate were ...   power to detect it:")
    for true_rate, pwr in r["power_curve"]:
        flag = "  <-- adequately powered" if pwr >= 0.8 else ""
        print(f"    {int(true_rate*100)}%                          {pwr:.2f}{flag}")
    print("\nLesson: with only ~40 shootouts, only a LARGE advantage (>~70%)")
    print("could be reliably detected. Absence of significance here is as much")
    print("about limited power as about the true effect — and saying so is the")
    print("difference between a careful analyst and a careless one.")