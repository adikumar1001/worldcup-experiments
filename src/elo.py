"""
elo.py
------
A simple, well-understood Elo rating system calibrated on historical World
Cup match results. Used by Module 3 to give simulated teams realistic,
heterogeneous strengths (a simulation where every team is equal would make
the format comparison meaningless).

Elo basics
    expected_score(A) = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    R_A' = R_A + K * (actual - expected)
We process all men's matches chronologically, updating ratings after each.
The spread of final ratings becomes the population we sample team strengths
from when generating virtual tournaments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_matches

BASE_RATING = 1500.00
K_FACTOR = 40.0

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0/(1.0 + 10** ((r_b - r_a)/400.0))

def compute_ratings() -> pd.Series:
     """Return final Elo rating per team after processing all men's matches."""
     m=load_matches()
     m=m[m.is_mens].dropna(subset=["match_date"]).sort_values("match_date")

     ratings: dict[str, float] = {}

     def get(team):
          return ratings.get(team, BASE_RATING)
     

     for _, row in m.iterrows():
          h, a= row["home_team_name"], row["away_team_name"]
          rh, ra = get(h), get(a)
          eh = expected_score(rh, ra)
          ea= 1- eh

            # Actual result (use full-time score; shootouts count as a draw for Elo)

          if row["home_team_score"]> row ["away_team_score"]:
              sh, sa = 1.0, 0.0

          elif row["away_team_score"] > row["home_team_score"]:
               sh, sa = 0.0, 1.0
          else:
               sh, sa = 0.5, 0.5

          ratings[h] =  rh + K_FACTOR*(sh-eh)
          ratings[a] = ra + K_FACTOR*(sa-ea)
    
     return pd.Series(ratings).sort_values(ascending=False) 

def strength_distribution(n: int=48, seed:int | None= None) -> np.ndarray:
     """
     Draw `n` team strengths to populate a virtual tournament. We sample from
     the empirical distribution of real final Elo ratings so the simulated
     field has a realistic spread of strong and weak sides.
     """

     rng =np.random.default_rng(seed)
     ratings = compute_ratings().values
     # Sample with replacement from real ratings, then jitter slightly.
     drawn = rng.choice(ratings, size=n, replace=True)
     drawn = drawn + rng.normal(0, 15, size=n)
     return np.sort(drawn)[::-1]  # strongest first


if __name__ == "__main__":
    r = compute_ratings()
    print("Top 10 all-time World Cup Elo (men's):")
    print(r.head(10).round(0).to_string())
    print("\nBottom 5:")
    print(r.tail(5).round(0).to_string())
    print(f"\nMean {r.mean():.0f}  |  SD {r.std():.0f}  |  range "
          f"{r.min():.0f}-{r.max():.0f}")
