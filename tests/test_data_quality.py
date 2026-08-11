"""
Data-quality tests for the ingested SQLite database.

These skip automatically when the database is absent (it is gitignored — a
build artefact of data_ingestion.py, not source), so CI stays green on a fresh
clone while still running locally after ingestion.

Run after ingestion:  pytest tests/ -v
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "phonepe_pulse.db"

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="database not built — run `python data_ingestion.py` first",
)

EXPECTED_TABLES = {
    "agg_transactions": 5_174,
    "agg_users":        1_036,
    "agg_user_devices": 6_919,
    "top_entities":    19_133,
}


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(DB_PATH)
    yield c
    c.close()


@pytest.fixture(scope="module")
def txn(conn):
    return pd.read_sql("SELECT * FROM agg_transactions", conn)


# ── Schema and volume ─────────────────────────────────────────────────────────
class TestSchema:
    def test_all_tables_present(self, conn):
        found = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert set(EXPECTED_TABLES).issubset(found)

    @pytest.mark.parametrize("table,expected", EXPECTED_TABLES.items())
    def test_row_counts_stable(self, conn, table, expected):
        """Guards against a partial or duplicated ingestion run."""
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert n == expected, f"{table}: expected {expected} rows, found {n}"

    def test_no_duplicate_transaction_rows(self, txn):
        key = ["year", "quarter", "state", "transaction_type"]
        assert not txn.duplicated(subset=key).any()


# ── Nulls and ranges ──────────────────────────────────────────────────────────
class TestIntegrity:
    def test_no_nulls_in_key_columns(self, txn):
        for col in ["year", "quarter", "state", "transaction_type",
                    "txn_count", "txn_amount"]:
            assert txn[col].notna().all(), f"nulls found in {col}"

    def test_quarters_in_range(self, txn):
        assert txn["quarter"].between(1, 4).all()

    def test_years_in_range(self, txn):
        assert txn["year"].between(2018, 2100).all()

    def test_no_negative_values(self, txn):
        assert (txn["txn_count"] >= 0).all()
        assert (txn["txn_amount"] >= 0).all()

    def test_state_count(self, txn):
        """36 states/UTs plus the 'india' national roll-up."""
        assert txn["state"].nunique() == 37
        assert "india" in txn["state"].values

    def test_average_ticket_is_plausible(self, txn):
        """Catches unit errors — amounts in paise instead of rupees would put
        every ticket 100x out. Floor is ₹10 rather than ₹50 because three
        genuine Financial Services rows in tiny UTs sit near ₹17–35 on
        transaction counts of 2, 42 and 27,970."""
        d = txn[txn["txn_count"] > 0]
        ticket = d["txn_amount"] / d["txn_count"]
        assert ticket.between(10, 50_000).all(), (
            f"min ₹{ticket.min():.2f}, max ₹{ticket.max():.2f}"
        )

    def test_high_volume_tickets_are_tight(self, txn):
        """Rows carrying real volume should sit in a narrow, sane band — this is
        the assertion that would actually catch a unit or parsing regression."""
        d = txn[txn["txn_count"] > 100_000]
        ticket = d["txn_amount"] / d["txn_count"]
        assert ticket.between(50, 20_000).all(), (
            f"min ₹{ticket.min():.2f}, max ₹{ticket.max():.2f}"
        )


# ── Cross-table reconciliation ────────────────────────────────────────────────
class TestReconciliation:
    def test_states_sum_close_to_national(self, txn):
        """State rows should reconcile with the 'india' roll-up. They will not
        match exactly — PhonePe publishes them separately — but a gap beyond a
        few percent means the ingestion dropped or double-counted a region."""
        nat = txn[txn.state == "india"]["txn_count"].sum()
        sts = txn[txn.state != "india"]["txn_count"].sum()
        assert abs(sts - nat) / nat < 0.05, f"national {nat:,} vs states {sts:,}"

    def test_every_state_has_population(self, txn):
        import analytics as A
        pop = set(A.load_population()["state"])
        states = set(txn[txn.state != "india"]["state"])
        assert states == pop, f"mismatch: {states ^ pop}"

    def test_users_cover_same_states(self, conn, txn):
        users = pd.read_sql("SELECT DISTINCT state FROM agg_users", conn)
        assert set(users["state"]) == set(txn["state"])


# ── Taxonomy stability ────────────────────────────────────────────────────────
class TestTaxonomy:
    def test_transaction_types_known(self, txn):
        """A new or renamed category silently breaks every mix chart."""
        expected = {
            "Merchant payments", "Peer-to-peer payments",
            "Recharge & bill payments", "Financial Services", "Others",
        }
        assert set(txn["transaction_type"]) == expected

    def test_every_year_has_every_type(self, txn):
        """The Financial Services series dips oddly in 2020-22. This asserts the
        category never disappears outright, which would indicate a taxonomy
        change rather than a behavioural shift."""
        nat = txn[txn.state == "india"]
        counts = nat.groupby(["year", "transaction_type"]).size().unstack(fill_value=0)
        assert (counts > 0).all().all(), f"missing category-years:\n{counts}"

    def test_entity_levels_known(self, conn):
        levels = {r[0] for r in conn.execute(
            "SELECT DISTINCT entity_level FROM top_entities")}
        assert levels == {"state", "district", "pincode"}
