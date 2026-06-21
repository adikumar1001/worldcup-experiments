"""
data_loader.py
--------------
Downloads (and caches) the jfjelstul/worldcup datasets, then provides
cleaned, analysis-ready frames used by all three experiment modules.

Source: https://github.com/jfjelstul/worldcup  (open data)
"""
from __future__ import annotations

import os
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
FILES = ["matches", "goals", "bookings", "penalty_kicks"]



# Resolve the data directory relative to this file, so it works from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE,"..","DATA"))

# VAR was introduced at the men's tournament held in 2018.
VAR_YEAR = 2018

#dOWNLOAD/ CACHE

def _path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.csv")

def download(force: bool= False) -> None:
     """Fetch each CSV into DATA_DIR unless already cached."""
     os.makedirs(DATA_DIR, exist_ok=True)
     for name in FILES:
          dest = _path(name)
          if os.path.exists(dest) and not force:
               continue
          url = f"{BASE_URL}/{name}.csv"
          print(f"Downloading {url}")
          pd.read_csv(url).to_csv(dest, index=False)


def _load(name: str) -> pd.DataFrame:
     dest= _path(name)
     if not os.path.exists(dest):
          download()
     return pd.read_csv(dest)

# --------------------------------------------------------------------------- #
# Cleaned accessors
# --------------------------------------------------------------------------- #

def load_matches() -> pd.DataFrame:
    """All World Cup matches with a parsed `year` and a men's/women's flag."""
    m = _load("matches")
    m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce")
    m["year"] = m["match_date"].dt.year
    m["is_mens"] = ~m["tournament_name"].str.contains("Women", case=False, na=False)
    return m

def load_goals() -> pd.DataFrame:
    g = _load("goals")
    g["match_date"] = pd.to_datetime(g["match_date"], errors="coerce")
    g["year"] = g["match_date"].dt.year
    g["is_mens"] = ~g["tournament_name"].str.contains("Women", case=False, na=False)
    return g


def load_bookings() -> pd.DataFrame:
    b = _load("bookings")
    b["match_date"] = pd.to_datetime(b["match_date"], errors="coerce")
    b["year"] = b["match_date"].dt.year
    b["is_mens"] = ~b["tournament_name"].str.contains("Women", case=False, na=False)
    return b

def load_penalty_kicks() ->pd.DataFrame:
     """Shootout kicks, ordered by `key_id` so kick sequence is preserved."""
     pk = _load("penalty_kicks")
     return pk.sort_values("key_id").reset_index(drop=True)
# --------------------------------------------------------------------------- #
# Derived: per-tournament rate table (Module 1 input)
# --------------------------------------------------------------------------- #

def tournament_rates(mens_only: bool= True) -> pd.Dataframe:
     """
    One row per tournament with penalty-goal rate and yellow-card rate
    per match. This is the panel the interrupted time-series consumes.
    """
     
     m, g, b = load_matches(), load_goals(), load_bookings()
     if mens_only:
          m, g,b = m[m.is_mens], g[g.is_mens], b[b.is_mens]
     matches = m.groupby("tournament_name").agg(year=("year", "max"), matches=("match_id", "nunique"))
     pens= (
          g[g["penalty"]==1].groupby("tournament_name").size().rename("penalty_goals")

    
     )
     yellows = (
        b[b["yellow_card"] == 1].groupby("tournament_name").size().rename("yellows")
     )
     reds = (
        b[b["red_card"] == 1].groupby("tournament_name").size().rename("reds")
     )
     
     t= matches.join([pens, yellows, reds]).reset_index().fillna(0)
     t["penalty_rate"] = t["penalty_goals"]/t["matches"]
     t["yellow_rate"] = t["yellows"]/t["matches"]
     t["red_rate"] = t["reds"] / t["matches"]
     t["post_var"] = (t["year"]>= VAR_YEAR).astype(int)
     t["years_since_first"] = t["year"] - t["year"].min()
     return t.sort_values("year").reset_index(drop=True)

if __name__ == "__main__":
    download()
    rates = tournament_rates(mens_only=True)
    print(rates[["tournament_name", "year", "matches",
                 "penalty_rate", "yellow_rate", "post_var"]].to_string(index=False))


