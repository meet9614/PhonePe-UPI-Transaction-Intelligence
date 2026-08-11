"""
analytics.py — analytical models for the PhonePe UPI project.
=============================================================
Pure functions, no Streamlit imports, so they can be unit-tested
independently of the dashboard.

Contents
--------
load_population()          : Census-2011 populations, reorganisation-adjusted
add_per_capita()           : transactions per person
merchant_adjusted_outliers(): anomaly detection that controls for merchant mix
fit_adoption_curve()       : logistic S-curve fit -> quarters behind the leader
normalise_state()          : PhonePe slug -> canonical display name
GEOJSON_ALIASES            : PhonePe slug -> India GeoJSON ST_NM value
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
POP_PATH = BASE_DIR / "data" / "state_population.csv"

# India states GeoJSON (Plotly-compatible, keyed on properties.ST_NM)
GEOJSON_URL = (
    "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/"
    "raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
)

# PhonePe slugs whose GeoJSON ST_NM spelling differs from a naive title-case.
# Anything not listed here is resolved by normalise_state() + fuzzy fallback.
# Several of these encode *misspellings* in the upstream boundary file
# ('Dadara', 'Havelli', 'Arunanchal'). They are only applied when the name is
# actually present in the loaded GeoJSON, so a correctly-spelled file still
# resolves via the folded match in build_geo_lookup().
GEOJSON_ALIASES: dict[str, str] = {
    "andaman-&-nicobar-islands":          "Andaman & Nicobar",
    "dadra-&-nagar-haveli-&-daman-&-diu": "Dadara & Nagar Havelli",
    "delhi":                              "NCT of Delhi",
    "jammu-&-kashmir":                    "Jammu & Kashmir",
    "arunachal-pradesh":                  "Arunanchal Pradesh",
    "odisha":                             "Odisha",
    "puducherry":                         "Puducherry",
    "uttarakhand":                        "Uttarakhand",
    "telangana":                          "Telangana",
    "ladakh":                             "Ladakh",
}


# ── Naming ────────────────────────────────────────────────────────────────────
def normalise_state(slug: str) -> str:
    """'dadra-&-nagar-haveli' -> 'Dadra & Nagar Haveli'."""
    return slug.replace("-", " ").title().replace(" & ", " & ")


def geojson_key(slug: str) -> str:
    """Map a PhonePe slug to the GeoJSON ST_NM property value."""
    return GEOJSON_ALIASES.get(slug, normalise_state(slug))


def _fold(name: str) -> str:
    """Aggressively fold a place name for matching: lowercase, drop 'and'/'&',
    strip everything non-alphanumeric. 'Dadara & Nagar Havelli' and
    'dadra-&-nagar-haveli-&-daman-&-diu' both fold toward the same stem, which
    is what makes matching robust to the spelling drift between sources."""
    s = name.lower().replace("&", " ").replace("-", " ").replace("_", " ")
    s = s.replace(" and ", " ")
    for noise in ("nct of ", "state of ", "union territory of ", " islands", " island"):
        s = s.replace(noise, " ")
    return "".join(ch for ch in s if ch.isalnum())


def build_geo_lookup(geojson: dict, slugs, feature_key: str = "ST_NM") -> dict[str, str]:
    """
    Resolve PhonePe state slugs against whatever names a GeoJSON actually uses.

    Boundary files disagree with each other constantly ('Odisha' vs 'Orissa',
    'NCT of Delhi' vs 'Delhi', 'Dadara & Nagar Havelli' with two spelling
    errors in the official file). Rather than trusting a hardcoded alias table
    to be right about a file we cannot see at build time, this matches against
    the file at runtime in three passes — exact alias, folded exact, then
    folded prefix/containment — so it self-corrects.

    Returns {slug: geojson_name} for every slug it could resolve.
    """
    if not geojson:
        return {}
    names = [
        f.get("properties", {}).get(feature_key)
        for f in geojson.get("features", [])
    ]
    names = [n for n in names if n]
    folded = {_fold(n): n for n in names}

    lookup: dict[str, str] = {}
    for slug in slugs:
        alias = GEOJSON_ALIASES.get(slug)
        if alias and alias in names:                       # 1. trusted alias
            lookup[slug] = alias
            continue
        f = _fold(slug)
        if f in folded:                                    # 2. folded exact
            lookup[slug] = folded[f]
            continue
        cands = [                                          # 3. folded overlap
            orig for fold_name, orig in folded.items()
            if fold_name.startswith(f[:8]) or f.startswith(fold_name[:8])
        ]
        if len(cands) == 1:
            lookup[slug] = cands[0]
    return lookup


# ── Population / per-capita ───────────────────────────────────────────────────
def load_population(path: Path | str | None = None) -> pd.DataFrame:
    """Census 2011 populations, adjusted for the AP/Telangana, J&K/Ladakh and
    DNH-DD reorganisations so the 36 rows line up with PhonePe's state list."""
    df = pd.read_csv(path or POP_PATH)
    return df[["state", "population_2011"]]


def add_per_capita(
    df: pd.DataFrame,
    value_col: str = "txn_count",
    state_col: str = "state",
    pop: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach population and a per-capita column. Rows with no population match
    are kept with NaN rather than silently dropped."""
    pop = load_population() if pop is None else pop
    out = df.merge(pop, left_on=state_col, right_on="state", how="left")
    out["per_capita"] = out[value_col] / out["population_2011"]
    return out


# ── Anomaly detection that controls for merchant mix ──────────────────────────
def merchant_adjusted_outliers(
    state_df: pd.DataFrame,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    The naive approach z-scores average ticket size directly, which mostly
    rediscovers merchant-payment penetration (corr ~= -0.84 in this dataset):
    states that do fewer merchant payments mechanically show a larger average
    ticket, because P2P transfers are bigger than shop payments.

    This regresses average ticket on merchant share and z-scores the *residual*,
    so a state is only flagged if it is unusual relative to states with a
    comparable payment mix.

    Expects columns: state, txn_count, txn_amount, merchant_count.
    Returns the input plus avg_ticket, merchant_pct, expected_ticket,
    residual, residual_z and flag.
    """
    d = state_df.copy()
    d["avg_ticket"] = d["txn_amount"] / d["txn_count"]
    d["merchant_pct"] = 100 * d["merchant_count"] / d["txn_count"]

    # Ordinary least squares: avg_ticket ~ a + b * merchant_pct
    x, y = d["merchant_pct"].to_numpy(float), d["avg_ticket"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:                                   # not enough to regress
        d["expected_ticket"] = np.nan
        d["residual"] = np.nan
        d["residual_z"] = np.nan
        d["flag"] = "INSUFFICIENT DATA"
        return d

    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    d["expected_ticket"] = intercept + slope * d["merchant_pct"]
    d["residual"] = d["avg_ticket"] - d["expected_ticket"]

    sd = d["residual"].std(ddof=1)
    d["residual_z"] = 0.0 if not sd or np.isnan(sd) else d["residual"] / sd
    d["flag"] = np.select(
        [d["residual_z"] > threshold, d["residual_z"] < -threshold],
        ["ABOVE MODEL", "BELOW MODEL"],
        default="AS EXPECTED",
    )
    return d


def merchant_ticket_correlation(state_df: pd.DataFrame) -> float:
    """Pearson r between merchant share and average ticket — the relationship
    the naive z-score was accidentally measuring."""
    d = state_df.copy()
    d["avg_ticket"] = d["txn_amount"] / d["txn_count"]
    d["merchant_pct"] = 100 * d["merchant_count"] / d["txn_count"]
    return float(d["merchant_pct"].corr(d["avg_ticket"]))


# ── Logistic adoption curve ───────────────────────────────────────────────────
def fit_adoption_curve(
    users_df: pd.DataFrame,
    pop: pd.DataFrame | None = None,
    min_points: int = 6,
) -> pd.DataFrame:
    """
    Fit a logistic S-curve to each state's user penetration and report where it
    sits on that curve.

    Rather than fitting the 3-parameter logistic numerically, this linearises it:
    if p(t) = 1 / (1 + exp(-(a + b*t))) then logit(p) = a + b*t, so an OLS fit on
    the logit gives the growth rate b and the midpoint t0 = -a/b (the quarter at
    which a state reaches 50% of its saturation ceiling).

    Ranking states by t0 gives "quarters behind the leader" — a single number
    that explains the volume gaps, the merchant-mix gaps and the apparent
    small-state anomalies as one phenomenon: states are on the same curve at
    different points.

    Expects columns: state, year, quarter, registered_users.
    """
    pop = load_population() if pop is None else pop
    d = users_df[users_df["state"] != "india"].merge(pop, on="state", how="inner")
    d = d.sort_values(["state", "year", "quarter"])
    d["t"] = (d["year"] - d["year"].min()) * 4 + (d["quarter"] - 1)

    # Penetration, capped just below 1 so the logit stays finite. The ceiling is
    # >1 in practice because registered users include lapsed and duplicate
    # accounts, so we normalise against the observed maximum instead of 100%.
    d["penetration"] = d["registered_users"] / d["population_2011"]
    ceiling = d["penetration"].max() * 1.05

    rows = []
    for state, g in d.groupby("state"):
        p = (g["penetration"] / ceiling).clip(1e-4, 1 - 1e-4)
        t = g["t"].to_numpy(float)
        if len(g) < min_points or p.nunique() < 3:
            continue
        logit = np.log(p / (1 - p)).to_numpy(float)
        ok = np.isfinite(logit) & np.isfinite(t)
        if ok.sum() < min_points:
            continue
        b, a = np.polyfit(t[ok], logit[ok], 1)
        if b <= 0:                                     # not an adoption curve
            continue
        rows.append({
            "state":          state,
            "growth_rate":    b,
            "midpoint_q":     -a / b,
            "penetration_now": g["penetration"].iloc[-1],
            "n_points":       int(ok.sum()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("midpoint_q").reset_index(drop=True)
    out["quarters_behind_leader"] = (out["midpoint_q"] - out["midpoint_q"].min()).round(1)
    out["leader"] = out["state"].iloc[0]
    return out
