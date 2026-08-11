"""
Unit tests for analytics.py.

These run on synthetic data so they work in CI without the database, which is
gitignored (it is a build artefact of data_ingestion.py, not source).
"""

import numpy as np
import pandas as pd
import pytest

import analytics as A


# ── Name handling ─────────────────────────────────────────────────────────────
class TestNaming:
    def test_normalise_basic(self):
        assert A.normalise_state("tamil-nadu") == "Tamil Nadu"
        assert A.normalise_state("uttar-pradesh") == "Uttar Pradesh"

    def test_normalise_ampersand(self):
        assert A.normalise_state("jammu-&-kashmir") == "Jammu & Kashmir"

    def test_fold_collapses_spelling_variants(self):
        # The upstream GeoJSON misspells several names; folding must survive it.
        assert A._fold("Dadara & Nagar Havelli") == A._fold("dadara-nagar-havelli")
        assert A._fold("NCT of Delhi") == A._fold("delhi")
        assert A._fold("Andaman & Nicobar Islands") == A._fold("andaman-&-nicobar")

    @pytest.mark.parametrize("spelling", [
        ["Arunanchal Pradesh", "NCT of Delhi", "Dadara & Nagar Havelli", "Odisha"],
        ["Arunachal Pradesh", "Delhi", "Dadra & Nagar Haveli & Daman & Diu", "Odisha"],
    ])
    def test_geo_lookup_survives_both_spellings(self, spelling):
        """The matcher must resolve against a misspelled *and* a clean file."""
        gj = {"features": [{"properties": {"ST_NM": n}} for n in spelling]}
        slugs = ["arunachal-pradesh", "delhi",
                 "dadra-&-nagar-haveli-&-daman-&-diu", "odisha"]
        lookup = A.build_geo_lookup(gj, slugs)
        assert len(lookup) == 4, f"unresolved: {set(slugs) - set(lookup)}"

    def test_geo_lookup_empty_geojson(self):
        assert A.build_geo_lookup(None, ["kerala"]) == {}
        assert A.build_geo_lookup({}, ["kerala"]) == {}


# ── Population ────────────────────────────────────────────────────────────────
class TestPopulation:
    def test_covers_all_36_states(self):
        pop = A.load_population()
        assert len(pop) == 36
        assert pop["state"].is_unique

    def test_reconciles_to_census_total(self):
        """Reorganisation splits must not create or destroy people: the 36 rows
        should still sum to the 2011 national total."""
        pop = A.load_population()
        assert pop["population_2011"].sum() == pytest.approx(1_210_854_977, rel=0.001)

    def test_no_missing_or_zero(self):
        pop = A.load_population()
        assert pop["population_2011"].notna().all()
        assert (pop["population_2011"] > 0).all()

    def test_per_capita_keeps_unmatched_rows(self):
        """A state with no population row must survive as NaN, not vanish."""
        df = pd.DataFrame({"state": ["kerala", "atlantis"], "txn_count": [100, 100]})
        out = A.add_per_capita(df)
        assert len(out) == 2
        assert out.loc[out.state == "atlantis", "per_capita"].isna().all()


# ── Merchant-adjusted anomalies ───────────────────────────────────────────────
def _synthetic_states(n=30, seed=0, planted=None):
    """Build states where avg ticket is a clean linear function of merchant
    share, optionally planting one genuine outlier."""
    rng = np.random.default_rng(seed)
    merch_pct = rng.uniform(30, 70, n)
    avg_ticket = 2500 - 20 * merch_pct + rng.normal(0, 15, n)
    if planted is not None:
        avg_ticket[0] += planted
    count = rng.integers(1e6, 1e9, n)
    return pd.DataFrame({
        "state":          [f"state-{i}" for i in range(n)],
        "txn_count":      count,
        "txn_amount":     count * avg_ticket,
        "merchant_count": count * merch_pct / 100,
    })


class TestAnomalies:
    def test_recovers_known_correlation(self):
        df = _synthetic_states()
        assert A.merchant_ticket_correlation(df) < -0.9

    def test_flag_rate_is_calibrated(self):
        """On clean data the flag rate should track the threshold, not be zero:
        a ±1.5σ cut on a normal residual flags ~13% by construction. A rate far
        above that means the residuals aren't normalised properly."""
        res = A.merchant_adjusted_outliers(_synthetic_states(n=200, seed=7))
        rate = (res["flag"] != "AS EXPECTED").mean()
        assert 0.05 < rate < 0.25, f"flag rate {rate:.1%} off theoretical ~13.4%"

    def test_strict_threshold_flags_nothing_on_clean_data(self):
        res = A.merchant_adjusted_outliers(_synthetic_states(), threshold=4.0)
        assert (res["flag"] == "AS EXPECTED").all()

    def test_detects_planted_outlier(self):
        res = A.merchant_adjusted_outliers(_synthetic_states(planted=400))
        assert res.iloc[0]["flag"] == "ABOVE MODEL"
        assert (res["flag"] != "AS EXPECTED").sum() == 1

    def test_low_merchant_state_is_not_flagged(self):
        """The regression must NOT re-flag states that merely have low merchant
        share — that was the bug in the original z-score approach."""
        df = _synthetic_states(n=30)
        df.loc[0, "merchant_count"] = df.loc[0, "txn_count"] * 0.20   # very low
        df.loc[0, "txn_amount"] = df.loc[0, "txn_count"] * (2500 - 20 * 20)
        res = A.merchant_adjusted_outliers(df)
        assert res.iloc[0]["flag"] == "AS EXPECTED"

    def test_handles_too_few_rows(self):
        df = _synthetic_states(n=2)
        res = A.merchant_adjusted_outliers(df)
        assert (res["flag"] == "INSUFFICIENT DATA").all()

    def test_expected_ticket_is_finite(self):
        res = A.merchant_adjusted_outliers(_synthetic_states())
        assert np.isfinite(res["expected_ticket"]).all()


# ── Adoption curve ────────────────────────────────────────────────────────────
def _synthetic_users(midpoints, n_q=24):
    """Generate logistic user curves with known midpoints, so the fit can be
    checked against ground truth."""
    pop = A.load_population()
    states = pop["state"].tolist()[:len(midpoints)]
    rows = []
    for state, t0 in zip(states, midpoints):
        p_state = int(pop.loc[pop.state == state, "population_2011"].iloc[0])
        for t in range(n_q):
            pen = 1 / (1 + np.exp(-0.25 * (t - t0)))
            rows.append({
                "state": state,
                "year": 2018 + t // 4,
                "quarter": t % 4 + 1,
                "registered_users": int(pen * p_state * 0.5),
            })
    return pd.DataFrame(rows)


class TestAdoptionCurve:
    def test_recovers_midpoint_ordering(self):
        """A state with an earlier true midpoint must rank ahead."""
        df = _synthetic_users([4, 10, 16])
        out = A.fit_adoption_curve(df)
        assert len(out) == 3
        assert out["midpoint_q"].is_monotonic_increasing

    def test_leader_has_zero_gap(self):
        out = A.fit_adoption_curve(_synthetic_users([4, 10, 16]))
        assert out.iloc[0]["quarters_behind_leader"] == 0

    def test_gap_is_positive_and_ordered(self):
        out = A.fit_adoption_curve(_synthetic_users([4, 10, 16]))
        assert (out["quarters_behind_leader"] >= 0).all()
        assert out["quarters_behind_leader"].is_monotonic_increasing

    def test_ignores_national_row(self):
        df = _synthetic_users([4, 10])
        national = df.copy()
        national["state"] = "india"
        out = A.fit_adoption_curve(pd.concat([df, national]))
        assert "india" not in out["state"].values

    def test_too_few_points_returns_empty(self):
        out = A.fit_adoption_curve(_synthetic_users([4], n_q=3))
        assert out.empty

    def test_growth_rate_positive(self):
        out = A.fit_adoption_curve(_synthetic_users([4, 10, 16]))
        assert (out["growth_rate"] > 0).all()
