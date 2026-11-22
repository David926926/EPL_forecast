"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database.sqlite"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MATCHES_PATH = PROCESSED / "matches.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"
MARKET_PATH = PROCESSED / "market.parquet"

EPL_LEAGUE_ID = 1729
SEASONS = [f"{y}/{y + 1}" for y in range(2008, 2016)]
WARMUP_SEASON = "2008/2009"

# Outcome classes, ordered for RPS: home does best -> worst.
CLASSES = ["H", "D", "A"]

# Bookmakers that are (nearly) complete in all eight seasons; PS is missing 2008/09-2011/12.
CONSENSUS_BOOKS = ["B365", "BW", "IW", "LB", "WH", "VC"]

# 2008/09 promoted clubs: the database starts in 2008/09 so it cannot tell us who came up
# that season. Source: Premier League history (West Brom, Stoke, Hull won promotion in 2007/08).
PROMOTED_2008 = {"West Bromwich Albion", "Stoke City", "Hull City"}

# Rolling-origin backtest folds (plan section 4).
FOLDS = [
    {"fold": 1, "train": ["2009/2010", "2010/2011"], "val": "2011/2012", "test": "2012/2013"},
    {"fold": 2, "train": ["2009/2010", "2010/2011", "2011/2012"], "val": "2012/2013", "test": "2013/2014"},
    {"fold": 3, "train": ["2009/2010", "2010/2011", "2011/2012", "2012/2013"], "val": "2013/2014", "test": "2014/2015"},
    {"fold": 4, "train": ["2009/2010", "2010/2011", "2011/2012", "2012/2013", "2013/2014"], "val": "2014/2015", "test": "2015/2016"},
]
