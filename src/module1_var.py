"""
module1_var.py
--------------
MODULE 1 — Did VAR change the game? (Interrupted Time-Series)

Question
    Did the introduction of Video Assistant Referee (VAR) at the 2018 men's
    World Cup cause a statistically significant change in the penalty-goal
    rate, after accounting for any pre-existing trend?

Why this is causal-flavoured, not just a before/after average
    A naive "mean before vs mean after" comparison would attribute *all* of
    any difference to VAR, even drift that was already happening. An
    interrupted time-series (ITS) regression models the underlying trend
    explicitly and tests whether there is a discrete LEVEL SHIFT at the exact
    moment of the intervention (2018) on top of that trend. The coefficient
    on the `post_var` indicator is the estimated causal effect under the ITS
    identifying assumption (no other change coincided exactly with 2018).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data_loader import tournament_rates, VAR_YEAR

def fit_its(outcome: str="penalty_rate") -> dict:
        """Fit the interrupted time-series model for a given outcome rate."""

        t=tournament_rates(mens_only=True)
        # Yellow/red cards only exist from 1970 onward; restrict to a valid window
     # so the trend term isn't biased by structural zeros.
        if outcome in("yellow_rate", "red_rate"):
         t=t[t["year"] >= 1970].copy()
        t["years_since_first"]= t["year"]-t["year"].min()

        model =smf.ols(f"{outcome} ~ years_since_first + post_var", data = t).fit()
      
        coef = model.params["post_var"]
        pval=model.pvalues["post_var"]
        ci_low, ci_high= model.conf_int().loc["post_var"]

        return {
             "outcome":outcome,
             "n_tournaments": len(t),
             "trend_per_tournament": model.params["years_since_first"],
             "var_effect":coef,
             "p_value": pval,
             "ci_95":(ci_low,ci_high),
             "model": model,
             "data": t,
            
      }




def placebo_test(outcome: str= "penalty_rate", fake_year:int=2006)-> dict:
        """
        Robustness check: pretend the intervention happened at `fake_year`
        (a year with no VAR). A credible result shows NO significant jump here.
        """
        t= tournament_rates(mens_only=True)
        if outcome in ("yellow_rate", "red_rate"):
            t = t[t["year"]>=1970].copy()
        t["years_since_first"] = t["year"] -t["year"].min()
      # Exclude the real post-VAR era so the placebo is tested on clean ground.
        t = t[t["year"] < VAR_YEAR].copy()
        t["fake_post"] = (t["year"] >= fake_year).astype(int) 
      
        model=smf.ols(f"{outcome} ~ years_since_first + fake_post", data=t).fit()
        return{
            "fake_year": fake_year,
            "fake_effect": model.params["fake_post"],
            "p_value": model.pvalues["fake_post"],
      }

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 1 — VAR Interrupted Time-Series (men's World Cups)")
    print("=" * 70)

    main = fit_its("penalty_rate")
    print(f"\nPenalty-goal rate per match")
    print(f"  Pre-VAR trend       : {main['trend_per_tournament']:+.5f} per tournament")
    print(f"  VAR level shift     : {main['var_effect']:+.4f} penalties/match")
    print(f"  95% CI              : [{main['ci_95'][0]:.4f}, {main['ci_95'][1]:.4f}]")
    print(f"  p-value             : {main['p_value']:.4f}")
    sig = "STATISTICALLY SIGNIFICANT" if main["p_value"] < 0.05 else "not significant"
    print(f"  Verdict             : {sig} at alpha=0.05")

    placebo = placebo_test("penalty_rate", fake_year=2006)
    print(f"\nPlacebo check (fake intervention at {placebo['fake_year']}):")
    print(f"  Fake effect         : {placebo['fake_effect']:+.4f} (p={placebo['p_value']:.4f})")