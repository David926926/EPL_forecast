"""Step 2: pre-match features -> data/processed/features.parquet.

Every feature uses only information available before kick-off. "diff" = home minus away.
Run after build_matches: python -m src.features

Feature layers (nested, from the most basic to the most situational):
  F1 player ability  = FIFA ratings of the starting XI by position (GK / DEF / MID / ATT), home - away;
  F2 = F1 + team performance = rolling 38-match (about one season) goal difference and shots on target
       per match, plus promoted flags (what ratings miss: tactics, coach, cohesion, depth, rating error);
  F3 = F2 + recent / situational = 5-match form, goal environment, h2h, rest, season stage, and how
       today's XI compares with the usual XI.
  T  = team performance only (the reverse-order reference: ratings added to T gives "F2 vs T").

Design choices not fixed by the plan (documented here and in README):
- Team history window: a club's window spans its current EPL spell only, i.e. the current season
  plus earlier seasons it played in the EPL without a break. Old matches of returning clubs
  (e.g. Burnley 2009/10 -> 2014/15) are ignored as stale.
- Promoted clubs: when the window holds fewer than N matches, the empty slots are filled with
  the per-match average of promoted clubs in their first EPL season, using only seasons strictly
  before the current one.
- Player ratings: latest FIFA overall_rating with date strictly before the match date. A missing
  starter (80 ids across 78 matches) is skipped; a positional group left empty is filled with that
  side's XI mean.
"""
import numpy as np
import pandas as pd
import sqlite3

from src.config import DB_PATH, FEATURES_PATH, MATCHES_PATH, PROMOTED_2008, SEASONS

LONG_N = 38
FORM_N = 5
H2H_N = 5
XI_USUAL_N = 10
REST_CAP = 14

RATING_COLS = ["gk_diff", "def_diff", "mid_diff", "att_diff"]
TEAM_COLS = ["gd38_diff", "sot38_diff", "home_promoted", "away_promoted"]
CONTEXT_COLS = ["form_pts_diff", "form_gd_diff", "form_sot_diff", "goal_env", "h2h_pts_home", "h2h_missing",
                "rest_diff", "season_stage", "xi_vs_usual_diff"]
F1_COLS = RATING_COLS
F2_COLS = F1_COLS + TEAM_COLS
F3_COLS = F2_COLS + CONTEXT_COLS
BINARY_COLS = ["home_promoted", "away_promoted", "h2h_missing"]
GAP_COLS = ["abs_gd38_diff", "abs_xi_rating_diff", "abs_form_pts_diff"]
FEATURE_SETS = {"F1": F1_COLS, "F2": F2_COLS, "F3": F3_COLS, "T": TEAM_COLS}


# ---------------------------------------------------------------- team-match long table
def team_long(m):
    """One row per (match, team)."""
    parts = []
    for side, opp in [("home", "away"), ("away", "home")]:
        d = pd.DataFrame({
            "match_api_id": m.match_api_id, "date": m.date, "season": m.season,
            "side": side, "team": m[f"{side}_team"], "opp": m[f"{opp}_team"],
            "gf": m[f"{side}_team_goal"], "ga": m[f"{opp}_team_goal"],
            "sot": m[f"{side}_sot"], "shots": m[f"{side}_shots"],
        })
        parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    t["pts"] = np.select([t.gf > t.ga, t.gf == t.ga], [3, 1], 0)
    t["gd"] = t.gf - t.ga
    t["goals_total"] = t.gf + t.ga

    # season index, promoted flag, stint id
    si = {s: i for i, s in enumerate(SEASONS)}
    t["si"] = t.season.map(si)
    ts = t[["team", "si"]].drop_duplicates().sort_values(["team", "si"])
    ts["prev_si"] = ts.groupby("team").si.shift(1)
    ts["promoted"] = np.where(ts.si == 0, ts.team.isin(PROMOTED_2008), ts.prev_si != ts.si - 1).astype(int)
    ts["stint"] = ((ts.prev_si.isna()) | (ts.prev_si != ts.si - 1)).astype(int).groupby(ts.team).cumsum()
    t = t.merge(ts[["team", "si", "promoted", "stint"]], on=["team", "si"], how="left")
    return t.sort_values(["team", "date", "match_api_id"]).reset_index(drop=True)


def promoted_prior(t, stats):
    """prior[season][stat] = mean per-match stat of promoted clubs in their first EPL season,
    over seasons strictly before `season`. NaN for the first season."""
    first = t[t.promoted == 1]
    per_season = first.groupby("season")[stats].agg(["sum", "count"])
    prior = {}
    for i, s in enumerate(SEASONS):
        past = per_season.loc[per_season.index.isin(SEASONS[:i])]
        prior[s] = {c: (past[(c, "sum")].sum() / past[(c, "count")].sum()) if len(past) else np.nan
                    for c in stats}
    return prior


def rolling_form(t, stats, n, prior, prefix="form"):
    """Mean of the previous n matches within the team's stint, padded with the promoted prior.
    Output columns: f"{prefix}_{stat}" and f"{prefix}_{stat}_n" (matches actually observed)."""
    g = t.groupby(["team", "stint"])
    out = pd.DataFrame(index=t.index)
    for c in stats:
        prev = g[c].shift(1)
        roll_sum = prev.groupby([t.team, t.stint]).rolling(n, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        roll_cnt = prev.groupby([t.team, t.stint]).rolling(n, min_periods=1).count().reset_index(level=[0, 1], drop=True)
        roll_sum, roll_cnt = roll_sum.reindex(t.index).fillna(0), roll_cnt.reindex(t.index).fillna(0)
        p = t.season.map(lambda s: prior[s][c])
        padded = (roll_sum + p * (n - roll_cnt)) / n
        plain = roll_sum / roll_cnt.replace(0, np.nan)
        out[f"{prefix}_{c}"] = np.where(p.notna(), padded, plain)
        out[f"{prefix}_{c}_n"] = roll_cnt
    return out


# ---------------------------------------------------------------- recent / situational (layer F3)
def head_to_head(m, fill):
    hist, pts_out, n_out = {}, [], []
    for r in m.itertuples(index=False):
        key = frozenset((r.home_team, r.away_team))
        past = hist.get(key, [])[-H2H_N:]
        pts_out.append(np.mean([p[r.home_team] for p in past]) if past else fill)
        n_out.append(len(past))
        hp = 3 if r.home_team_goal > r.away_team_goal else (1 if r.home_team_goal == r.away_team_goal else 0)
        ap = 3 if hp == 0 else (1 if hp == 1 else 0)
        hist.setdefault(key, []).append({r.home_team: hp, r.away_team: ap})
    return np.array(pts_out), np.array(n_out)


def rest_days(t):
    prev = t.groupby("team").date.shift(1)
    return (t.date - prev).dt.days.fillna(REST_CAP).clip(upper=REST_CAP)


# ---------------------------------------------------------------- player ratings (layer F1)
def position_group(y):
    return np.select([y == 1, y == 3, (y >= 5) & (y <= 8), y >= 9], ["gk", "def", "mid", "att"], "other")


def lineup_ratings(m):
    """Per (match, side): mean rating by position group, XI mean, and rating staleness."""
    con = sqlite3.connect(DB_PATH)
    pa = pd.read_sql("SELECT player_api_id, date, overall_rating FROM Player_Attributes "
                     "WHERE overall_rating IS NOT NULL", con)
    con.close()
    pa["date"] = pd.to_datetime(pa.date)
    pa["rating_date"] = pa.date
    pa = pa.sort_values("date")

    slots = []
    for side in ["home", "away"]:
        for i in range(1, 12):
            slots.append(pd.DataFrame({"match_api_id": m.match_api_id, "date": m.date, "side": side,
                                       "player_api_id": m[f"{side}_player_{i}"],
                                       "y": m[f"{side}_player_Y{i}"]}))
    s = pd.concat(slots, ignore_index=True)
    s["pos"] = position_group(s.y.values)
    assert (s.pos != "other").all(), s.y[s.pos == "other"].unique()
    has_id = s.player_api_id.notna()
    rated = pd.merge_asof(
        s[has_id].astype({"player_api_id": "int64"}).sort_values("date"),
        pa[["player_api_id", "date", "rating_date", "overall_rating"]].astype({"player_api_id": "int64"}),
        on="date", by="player_api_id", direction="backward", allow_exact_matches=False)
    s = pd.concat([rated, s[~has_id]], ignore_index=True)

    s["rating_age_days"] = (s.date - s.rating_date).dt.days
    xi = s.groupby(["match_api_id", "side"]).agg(
        xi_rating=("overall_rating", "mean"), n_rated=("overall_rating", "count"),
        max_rating_age=("rating_age_days", "max"), min_rating_age=("rating_age_days", "min"))
    pos = s.pivot_table(index=["match_api_id", "side"], columns="pos", values="overall_rating", aggfunc="mean")
    xi = xi.join(pos[["gk", "def", "mid", "att"]])
    for c in ["gk", "def", "mid", "att"]:
        xi[c] = xi[c].fillna(xi.xi_rating)
    return xi.reset_index()


# ---------------------------------------------------------------- assemble
def build():
    m = pd.read_parquet(MATCHES_PATH).sort_values(["date", "match_api_id"]).reset_index(drop=True)
    f = m[["match_api_id", "season", "stage", "date", "home_team", "away_team", "y", "y_int"]].copy()

    # team-level rolling stats: long window (38, layer F2) and short window (5, layer F3)
    t = team_long(m)
    stats = ["pts", "gd", "sot", "shots", "goals_total"]
    prior = promoted_prior(t, stats)
    t = t.join(rolling_form(t, ["gd", "sot"], LONG_N, prior, prefix="r38"))
    t = t.join(rolling_form(t, stats, FORM_N, prior))
    t["rest"] = rest_days(t)

    xi = lineup_ratings(m)
    t = t.merge(xi, on=["match_api_id", "side"], how="left")
    t = t.sort_values(["team", "date", "match_api_id"]).reset_index(drop=True)
    prev_xi = t.groupby(["team", "stint"]).xi_rating.shift(1)
    usual = prev_xi.groupby([t.team, t.stint]).rolling(XI_USUAL_N, min_periods=1).mean() \
        .reset_index(level=[0, 1], drop=True).reindex(t.index)
    t["xi_vs_usual"] = (t.xi_rating - usual).fillna(0.0)

    cols = ["promoted", "r38_gd", "r38_sot", "r38_gd_n", "form_pts", "form_gd", "form_sot", "form_shots",
            "form_goals_total", "form_pts_n", "rest", "xi_rating", "gk", "def", "mid", "att", "xi_vs_usual",
            "n_rated", "max_rating_age", "min_rating_age"]
    wide = t.pivot(index="match_api_id", columns="side", values=cols)
    wide.columns = [f"{side}_{c}" for c, side in wide.columns]
    f = f.merge(wide.reset_index(), on="match_api_id", how="left")

    f["gd38_diff"] = f.home_r38_gd - f.away_r38_gd
    f["sot38_diff"] = f.home_r38_sot - f.away_r38_sot
    for c in ["pts", "gd", "sot", "shots"]:
        f[f"form_{c}_diff"] = f[f"home_form_{c}"] - f[f"away_form_{c}"]
    f["goal_env"] = f.home_form_goals_total + f.away_form_goals_total

    fill = t.loc[t.si == 0, "pts"].mean()  # neutral h2h fill: mean points per team-match in the warm-up season
    h2h_pts, h2h_n = head_to_head(m, fill)
    f["h2h_pts_home"], f["h2h_n"] = h2h_pts, h2h_n
    f["h2h_missing"] = (h2h_n == 0).astype(int)
    f["rest_diff"] = f.home_rest - f.away_rest
    f["season_stage"] = f.stage / 38

    f["xi_rating_diff"] = f.home_xi_rating - f.away_xi_rating
    for c in ["gk", "def", "mid", "att"]:
        f[f"{c}_diff"] = f[f"home_{c}"] - f[f"away_{c}"]
    f["xi_vs_usual_diff"] = f.home_xi_vs_usual - f.away_xi_vs_usual

    f["abs_gd38_diff"] = f.gd38_diff.abs()
    f["abs_xi_rating_diff"] = f.xi_rating_diff.abs()
    f["abs_form_pts_diff"] = f.form_pts_diff.abs()
    f["is_warmup"] = (f.season == SEASONS[0]).astype(int)
    for c in ["home_promoted", "away_promoted"]:
        f[c] = f[c].astype(int)
    return f


if __name__ == "__main__":
    f = build()
    f.to_parquet(FEATURES_PATH, index=False)
    print(f"saved {FEATURES_PATH} {f.shape}")
    model = f[f.is_warmup == 0]
    na = model[F3_COLS + GAP_COLS].isna().sum()
    print("NaN in model features (2009/10+):", na[na > 0].to_dict() or "none")
    print(model[F3_COLS + GAP_COLS].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
