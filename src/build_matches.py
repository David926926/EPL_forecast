"""Step 1: extract EPL matches from database.sqlite -> data/processed/matches.parquet.

One row per match: date, season, stage, teams, score, label y, 22 starters with X/Y
coordinates, bookmaker odds, and per-match event stats parsed from the XML columns.
The event stats are POST-match information: they may only be used in features through
lagged (previous-match) aggregates. Run: python -m src.build_matches
"""
import sqlite3
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.config import CLASSES, DB_PATH, EPL_LEAGUE_ID, MATCHES_PATH, PROCESSED, SEASONS

SIDES = ["home", "away"]
PLAYER_COLS = [f"{s}_player_{i}" for s in SIDES for i in range(1, 12)]
X_COLS = [f"{s}_player_X{i}" for s in SIDES for i in range(1, 12)]
Y_COLS = [f"{s}_player_Y{i}" for s in SIDES for i in range(1, 12)]
ODDS_BOOKS = ["B365", "BW", "IW", "LB", "PS", "WH", "VC"]
ODDS_COLS = [f"{b}{o}" for b in ODDS_BOOKS for o in "HDA"]


def _team_counts(xml, home_id, away_id, stat_tag=None):
    """Count XML events per team. If stat_tag is given, count only events whose
    <stats> contains that tag (e.g. 'shoton' vs 'blocked' inside the shoton column)."""
    h = a = 0
    if xml is None or len(xml) < 20:
        return np.nan, np.nan
    for v in ET.fromstring(xml).findall("value"):
        if stat_tag is not None:
            st = v.find("stats")
            if st is None or st.find(stat_tag) is None:
                continue
        team = v.findtext("team")
        if team is None:
            continue
        team = int(team)
        if team == home_id:
            h += 1
        elif team == away_id:
            a += 1
    return h, a


def _final_possession(xml):
    """Home possession % from the last possession snapshot in the match."""
    if xml is None or len(xml) < 20:
        return np.nan
    vals = [v.findtext("homepos") for v in ET.fromstring(xml).findall("value")]
    vals = [int(x) for x in vals if x is not None]
    return vals[-1] if vals else np.nan


def parse_events(m):
    rows = []
    for r in m.itertuples(index=False):
        h, a = r.home_team_api_id, r.away_team_api_id
        sot = _team_counts(r.shoton, h, a, "shoton")
        blk = _team_counts(r.shoton, h, a, "blocked")
        off = _team_counts(r.shotoff, h, a)
        cor = _team_counts(r.corner, h, a)
        rows.append({
            "home_sot": sot[0], "away_sot": sot[1],
            "home_blocked": blk[0], "away_blocked": blk[1],
            "home_shotoff": off[0], "away_shotoff": off[1],
            "home_corners": cor[0], "away_corners": cor[1],
            "home_possession": _final_possession(r.possession),
        })
    ev = pd.DataFrame(rows, index=m.index)
    # The XML 'shoton' events exclude goals (XML SOT barely rises with goals scored, and 5% of
    # one-goal team-matches have 0 XML SOT), so goals are added back. Own goals are a small overcount.
    for s in ["home", "away"]:
        ev[f"{s}_sot_xml"] = ev[f"{s}_sot"]
        ev[f"{s}_sot"] = ev[f"{s}_sot_xml"] + m[f"{s}_team_goal"]
        ev[f"{s}_shots"] = ev[f"{s}_sot"] + ev[f"{s}_blocked"] + ev[f"{s}_shotoff"]
    return ev


def build():
    con = sqlite3.connect(DB_PATH)
    m = pd.read_sql(
        "SELECT * FROM Match WHERE league_id = ? ORDER BY date, match_api_id",
        con, params=(EPL_LEAGUE_ID,),
    )
    teams = pd.read_sql("SELECT team_api_id, team_long_name FROM Team", con)
    con.close()

    name = teams.set_index("team_api_id").team_long_name
    m["date"] = pd.to_datetime(m["date"])
    m["home_team"] = m.home_team_api_id.map(name)
    m["away_team"] = m.away_team_api_id.map(name)
    m["y"] = np.select(
        [m.home_team_goal > m.away_team_goal, m.home_team_goal == m.away_team_goal],
        ["H", "D"], "A",
    )
    m["y_int"] = m.y.map({c: i for i, c in enumerate(CLASSES)})
    m["n_missing_starters"] = m[PLAYER_COLS].isna().sum(axis=1)

    ev = parse_events(m)
    keep = (["match_api_id", "season", "stage", "date", "home_team_api_id", "away_team_api_id",
             "home_team", "away_team", "home_team_goal", "away_team_goal", "y", "y_int",
             "n_missing_starters"] + PLAYER_COLS + X_COLS + Y_COLS + ODDS_COLS)
    out = pd.concat([m[keep], ev], axis=1).reset_index(drop=True)
    for c in PLAYER_COLS:
        out[c] = out[c].astype("Int64")
    validate(out)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MATCHES_PATH, index=False)
    out.loc[out.n_missing_starters > 0, ["season", "date", "home_team", "away_team", "n_missing_starters"]] \
        .to_csv(PROCESSED / "missing_starters.csv", index=False)
    return out


def validate(m):
    assert len(m) == 3040, len(m)
    assert m.match_api_id.is_unique
    assert (m.groupby("season").size() == 380).all()
    assert sorted(m.season.unique()) == SEASONS
    per_team = pd.concat([m[["season", "home_team"]].set_axis(["season", "team"], axis=1),
                          m[["season", "away_team"]].set_axis(["season", "team"], axis=1)])
    assert (per_team.groupby(["season", "team"]).size() == 38).all()
    assert (per_team.groupby("season").team.nunique() == 20).all()
    assert m[X_COLS + Y_COLS].notna().all().all()
    assert m.home_team.notna().all() and m.away_team.notna().all()


if __name__ == "__main__":
    out = build()
    print(f"saved {MATCHES_PATH} {out.shape}")
    print("label share by season:\n", pd.crosstab(out.season, out.y, normalize="index").round(3))
    print("matches with >=1 missing starter id:", (out.n_missing_starters > 0).sum(),
          "| missing ids total:", out.n_missing_starters.sum())
    print("event stats missing (home_sot NaN):", out.home_sot.isna().sum())
    print("mean per team per match: SOT %.2f, shots %.2f, corners %.2f" % (
        out[["home_sot", "away_sot"]].mean().mean(), out[["home_shots", "away_shots"]].mean().mean(),
        out[["home_corners", "away_corners"]].mean().mean()))
