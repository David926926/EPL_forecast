"""Data and leakage checkpoints. Every check is an assert; the script stops at the first
failure. Rolling features (38- and 5-match windows) and h2h are re-derived with a slow, independent
brute-force implementation and compared with the vectorised ones. Run: python -m src.checks
"""
import numpy as np
import pandas as pd

from src.config import FEATURES_PATH, MARKET_PATH, MATCHES_PATH, PROMOTED_2008, SEASONS
from src.features import F3_COLS, FORM_N, GAP_COLS, H2H_N, LONG_N

m = pd.read_parquet(MATCHES_PATH).sort_values(["date", "match_api_id"]).reset_index(drop=True)
f = pd.read_parquet(FEATURES_PATH).set_index("match_api_id").loc[m.match_api_id].reset_index()
mk = pd.read_parquet(MARKET_PATH)


def ok(msg):
    print("  OK ", msg)


print("[counts]")
assert len(m) == 3040 and len(f) == 3040 and len(mk) == 3040
team_games = pd.concat([m[["season", "home_team"]].set_axis(["s", "t"], axis=1),
                        m[["season", "away_team"]].set_axis(["s", "t"], axis=1)]).value_counts()
assert (team_games == 38).all()
ok("3040 matches, 38 per team-season")

print("[player ratings]")
assert (f[["home_min_rating_age", "away_min_rating_age"]] > 0).all().all()
ok(f"every rating snapshot is strictly before match date "
   f"(min age {int(f[['home_min_rating_age', 'away_min_rating_age']].min().min())} days, "
   f"median of per-XI max age {f.home_max_rating_age.median():.0f} days)")
n_ids = 22 - m.n_missing_starters
assert ((f.home_n_rated + f.away_n_rated) == n_ids).all()
ok("every starter with a player id got a rating")

print("[brute-force rolling features / h2h]")
rows = []
for side, opp in [("home", "away"), ("away", "home")]:
    rows.append(pd.DataFrame({"mid": m.match_api_id, "date": m.date, "season": m.season, "team": m[f"{side}_team"],
                              "gf": m[f"{side}_team_goal"], "ga": m[f"{opp}_team_goal"], "sot": m[f"{side}_sot"]}))
tl = pd.concat(rows, ignore_index=True)
tl["pts"] = np.select([tl.gf > tl.ga, tl.gf == tl.ga], [3, 1], 0)
tl["gd"] = tl.gf - tl.ga
in_season = {s: set(tl.team[tl.season == s]) for s in SEASONS}


def promoted(team, s):
    i = SEASONS.index(s)
    return team in PROMOTED_2008 if i == 0 else team not in in_season[SEASONS[i - 1]]


# prior[s][col]: per-match mean of promoted clubs in their first EPL season, seasons strictly before s
tl["first_season_promoted"] = [promoted(t, s) for t, s in zip(tl.team, tl.season)]
prior = {}
for i, s in enumerate(SEASONS):
    first = tl[tl.first_season_promoted & tl.season.isin(SEASONS[:i])]
    prior[s] = {c: (first[c].mean() if len(first) else np.nan) for c in ["pts", "gd", "sot"]}


def brute(team, date, s, col, n):
    """Mean of `col` over the team's previous n EPL matches (this season, plus last season if the club
    was in the EPL then), with missing slots filled by the promoted-club prior."""
    i = SEASONS.index(s)
    ok_seasons = [s] if promoted(team, s) or i == 0 else [SEASONS[i - 1], s]
    h = tl[(tl.team == team) & (tl.date < date) & tl.season.isin(ok_seasons)].sort_values("date").tail(n)
    p = prior[s][col]
    if np.isnan(p):
        return h[col].mean()
    return (h[col].sum() + p * (n - len(h))) / n


checks = [("form_pts", "pts", FORM_N), ("form_sot", "sot", FORM_N), ("r38_gd", "gd", LONG_N), ("r38_sot", "sot", LONG_N)]
rng = np.random.default_rng(0)
idx = rng.choice(len(m), 400, replace=False)
for j in idx:
    r = m.iloc[j]
    for side in ["home", "away"]:
        team = r[f"{side}_team"]
        for feat, col, n in checks:
            got, want = f[f"{side}_{feat}"].iloc[j], brute(team, r.date, r.season, col, n)
            assert np.isclose(got, want, equal_nan=True), (r.match_api_id, side, feat, got, want)
    past = m[(m.date < r.date) & (((m.home_team == r.home_team) & (m.away_team == r.away_team)) |
                                  ((m.home_team == r.away_team) & (m.away_team == r.home_team)))].tail(H2H_N)
    if len(past):
        hp = np.where(past.home_team == r.home_team,
                      np.select([past.home_team_goal > past.away_team_goal, past.home_team_goal == past.away_team_goal], [3, 1], 0),
                      np.select([past.away_team_goal > past.home_team_goal, past.home_team_goal == past.away_team_goal], [3, 1], 0))
        assert np.isclose(f.h2h_pts_home.iloc[j], hp.mean()), r.match_api_id
    assert f.h2h_n.iloc[j] == len(past)
ok("gd38 / sot38 (38-match) and form_pts / form_sot (5-match), both sides, plus h2h, match a brute-force "
   "recomputation on 400 random matches using only earlier matches")

model = f[f.season != SEASONS[0]]
for side in ["home", "away"]:
    non_prom = model[model[f"{side}_promoted"] == 0]
    assert (non_prom[f"{side}_r38_gd_n"] == LONG_N).all()
ok("from 2009/10 on, every non-promoted club has a full 38-match window")

print("[completeness / ranges]")
assert model[F3_COLS + GAP_COLS].notna().all().all()
ok("no NaN in F1-F3 / gap features for 2009/10+")
assert (model.groupby("season").home_promoted.sum() + model.groupby("season").away_promoted.sum() == 3 * 38).all()
ok("exactly 3 promoted clubs per season")
assert np.allclose(mk[["p_mkt_H", "p_mkt_D", "p_mkt_A"]].sum(axis=1), 1)
ok("market probabilities sum to 1")

print("[sanity]")
q = pd.qcut(model.abs_gd38_diff, 5, labels=["closest", "2", "3", "4", "widest"])
print("  draw rate by |gd38_diff| quintile:", model.groupby(q, observed=True).y.apply(lambda s: (s == "D").mean()).round(3).to_dict())
print("  home-win rate by gd38_diff quintile:",
      model.groupby(pd.qcut(model.gd38_diff, 5, labels=False), observed=True).y.apply(lambda s: (s == "H").mean()).round(3).to_dict())
print("  promoted prior (per-match goal diff of promoted clubs, by season):",
      {s: round(prior[s]["gd"], 3) for s in SEASONS[1:]})
assert (m.home_sot >= m.home_team_goal).all() and (m.away_sot >= m.away_team_goal).all()
print(f"  shots on target per team-match: {m[['home_sot', 'away_sot']].mean().mean():.2f} (goals added back to XML SOT)")
print("\nALL CHECKS PASSED")
