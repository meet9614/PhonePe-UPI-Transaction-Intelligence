"""
PhonePe Pulse Data Ingestion Pipeline
--------------------------------------
Parses all JSON files from the PhonePe/pulse GitHub repo and loads
them into a structured SQLite database ready for SQL analysis.

Usage:
    python data_ingestion.py --pulse_dir ./pulse --db_path ./data/phonepe_pulse.db

Tables created:
    agg_transactions   — Year/quarter/state/type-level transaction counts & amounts
    agg_users          — Year/quarter/state-level registered users & app opens
    agg_user_devices   — Year/quarter/state-level device brand breakdown
    top_entities       — Top states, districts, pincodes per quarter
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS agg_transactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    year              INTEGER NOT NULL,
    quarter           INTEGER NOT NULL,
    state             TEXT    NOT NULL DEFAULT 'india',   -- 'india' = national
    transaction_type  TEXT    NOT NULL,
    txn_count         INTEGER NOT NULL,
    txn_amount        REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS agg_users (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    year              INTEGER NOT NULL,
    quarter           INTEGER NOT NULL,
    state             TEXT    NOT NULL DEFAULT 'india',
    registered_users  INTEGER NOT NULL,
    app_opens         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agg_user_devices (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    year              INTEGER NOT NULL,
    quarter           INTEGER NOT NULL,
    state             TEXT    NOT NULL DEFAULT 'india',
    brand             TEXT    NOT NULL,
    user_count        INTEGER NOT NULL,
    percentage        REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS top_entities (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    year              INTEGER NOT NULL,
    quarter           INTEGER NOT NULL,
    state             TEXT    NOT NULL DEFAULT 'india',
    entity_level      TEXT    NOT NULL,   -- 'state', 'district', 'pincode'
    entity_name       TEXT    NOT NULL,
    txn_count         INTEGER NOT NULL,
    txn_amount        REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agg_txn_ysq  ON agg_transactions (year, quarter, state);
CREATE INDEX IF NOT EXISTS idx_agg_usr_ysq  ON agg_users         (year, quarter, state);
CREATE INDEX IF NOT EXISTS idx_top_ent_ysq  ON top_entities       (year, quarter, state, entity_level);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _open_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_year_quarters(base: Path):
    """Yield (year, quarter, path) for all {year}/{quarter}.json files."""
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        for qfile in sorted(year_dir.iterdir()):
            if qfile.suffix == ".json" and qfile.stem.isdigit():
                yield year, int(qfile.stem), qfile


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_agg_transactions(cursor, pulse_dir: Path):
    """Load aggregated/transaction — country + all states."""
    rows = []

    def _parse(year, quarter, state, path):
        data = _open_json(path)
        if not data.get("success"):
            return
        for txn in data["data"].get("transactionData", []):
            instr = txn["paymentInstruments"][0]
            rows.append((year, quarter, state,
                         txn["name"], instr["count"], instr["amount"]))

    # Country level
    country_base = pulse_dir / "data/aggregated/transaction/country/india"
    for year, quarter, path in _iter_year_quarters(country_base):
        _parse(year, quarter, "india", path)

    # State level
    state_base = country_base / "state"
    for state_dir in sorted(state_base.iterdir()):
        if not state_dir.is_dir():
            continue
        for year, quarter, path in _iter_year_quarters(state_dir):
            _parse(year, quarter, state_dir.name, path)

    cursor.executemany(
        "INSERT INTO agg_transactions (year,quarter,state,transaction_type,txn_count,txn_amount) "
        "VALUES (?,?,?,?,?,?)", rows
    )
    log.info(f"agg_transactions: {len(rows):,} rows inserted")


def load_agg_users(cursor, pulse_dir: Path):
    """Load aggregated/user — country + all states, plus device breakdown."""
    user_rows, device_rows = [], []

    def _parse(year, quarter, state, path):
        data = _open_json(path)
        if not data.get("success"):
            return
        agg = data["data"].get("aggregated", {})
        user_rows.append((year, quarter, state,
                          agg.get("registeredUsers", 0),
                          agg.get("appOpens", 0)))
        for dev in (data["data"].get("usersByDevice") or []):
            device_rows.append((year, quarter, state,
                                dev["brand"], dev["count"], dev["percentage"]))

    country_base = pulse_dir / "data/aggregated/user/country/india"
    for year, quarter, path in _iter_year_quarters(country_base):
        _parse(year, quarter, "india", path)

    state_base = country_base / "state"
    for state_dir in sorted(state_base.iterdir()):
        if not state_dir.is_dir():
            continue
        for year, quarter, path in _iter_year_quarters(state_dir):
            _parse(year, quarter, state_dir.name, path)

    cursor.executemany(
        "INSERT INTO agg_users (year,quarter,state,registered_users,app_opens) "
        "VALUES (?,?,?,?,?)", user_rows
    )
    cursor.executemany(
        "INSERT INTO agg_user_devices (year,quarter,state,brand,user_count,percentage) "
        "VALUES (?,?,?,?,?,?)", device_rows
    )
    log.info(f"agg_users: {len(user_rows):,} rows | agg_user_devices: {len(device_rows):,} rows")


def load_top_entities(cursor, pulse_dir: Path):
    """Load top/transaction — country + state, states/districts/pincodes."""
    rows = []

    def _parse(year, quarter, state, path):
        data = _open_json(path)
        if not data.get("success"):
            return
        for level in ("states", "districts", "pincodes"):
            for ent in (data["data"].get(level) or []):
                if not ent.get("entityName"):   # skip rare null-name rows in Ladakh early data
                    continue
                m = ent["metric"]
                rows.append((year, quarter, state,
                             level.rstrip("s"),   # 'state', 'district', 'pincode'
                             ent["entityName"], m["count"], m["amount"]))

    country_base = pulse_dir / "data/top/transaction/country/india"
    for year, quarter, path in _iter_year_quarters(country_base):
        _parse(year, quarter, "india", path)

    state_base = country_base / "state"
    for state_dir in sorted(state_base.iterdir()):
        if not state_dir.is_dir():
            continue
        for year, quarter, path in _iter_year_quarters(state_dir):
            _parse(year, quarter, state_dir.name, path)

    cursor.executemany(
        "INSERT INTO top_entities (year,quarter,state,entity_level,entity_name,txn_count,txn_amount) "
        "VALUES (?,?,?,?,?,?,?)", rows
    )
    log.info(f"top_entities: {len(rows):,} rows inserted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_database(pulse_dir: str, db_path: str):
    pulse_dir = Path(pulse_dir)
    db_path   = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()
        log.info("Removed existing database.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    log.info("Creating schema...")
    cursor.executescript(SCHEMA)

    log.info("Loading aggregated transactions...")
    load_agg_transactions(cursor, pulse_dir)

    log.info("Loading aggregated users...")
    load_agg_users(cursor, pulse_dir)

    log.info("Loading top entities...")
    load_top_entities(cursor, pulse_dir)

    conn.commit()
    conn.close()
    log.info(f"\n✅ Database built: {db_path}  ({db_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PhonePe Pulse data into SQLite")
    parser.add_argument("--pulse_dir", default="./pulse",
                        help="Path to cloned PhonePe/pulse repo")
    parser.add_argument("--db_path", default="./data/phonepe_pulse.db",
                        help="Output SQLite database path")
    args = parser.parse_args()
    build_database(args.pulse_dir, args.db_path)
