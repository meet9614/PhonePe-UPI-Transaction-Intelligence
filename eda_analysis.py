"""
PhonePe UPI Transaction Intelligence — EDA & Excel Export
==========================================================
Run as a script:  python eda_analysis.py
Or convert to notebook: jupytext --to notebook eda_analysis.py

Outputs:
  assets/eda_plots/   — 8 PNG visualisation files
  data/phonepe_summary.xlsx — formatted Excel workbook with pivot tables
"""

import sqlite3
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from analytics import (
    merchant_adjusted_outliers,
    merchant_ticket_correlation,
    normalise_state,
)

warnings.filterwarnings("ignore")

# ── Style ───────────────────────────────────────────────────────────────────
PHONEPE_PURPLE = "#5f259f"
PHONEPE_GREEN  = "#00b9f5"
ACCENT         = "#e8453c"
BG             = "#f8f8f8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

BASE_DIR  = Path(__file__).resolve().parent
DB_PATH   = BASE_DIR / "data" / "phonepe_pulse.db"
OUT_DIR   = BASE_DIR / "assets" / "eda_plots"
XLSX_PATH = BASE_DIR / "data" / "phonepe_summary.xlsx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DB_PATH.exists():
    raise SystemExit(
        f"Database not found at {DB_PATH}\n"
        "Run `python data_ingestion.py` from the project folder first."
    )

conn = sqlite3.connect(DB_PATH)

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ── Load core dataframes ─────────────────────────────────────────────────────
df_txn   = pd.read_sql("SELECT * FROM agg_transactions", conn)
df_users = pd.read_sql("SELECT * FROM agg_users", conn)
df_top   = pd.read_sql("SELECT * FROM top_entities", conn)

df_national = df_txn[df_txn["state"] == "india"].copy()
df_states   = df_txn[df_txn["state"] != "india"].copy()

# ── Plot 1: Total UPI Transaction Volume — Quarterly bar chart ───────────────
print("\n[1/8] Quarterly transaction volume...")
q_vol = (df_national.groupby(["year","quarter"])["txn_count"]
         .sum().reset_index())
q_vol["period"] = q_vol["year"].astype(str) + "-Q" + q_vol["quarter"].astype(str)
q_vol["txns_bn"] = q_vol["txn_count"] / 1e9

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(q_vol["period"], q_vol["txns_bn"], color=PHONEPE_PURPLE, alpha=0.85, width=0.7)
ax.set_title("PhonePe UPI — Total Transaction Volume per Quarter (2018–2024)")
ax.set_ylabel("Transactions (Billions)")
ax.set_xlabel("")
ax.set_xticks(range(len(q_vol)))
ax.set_xticklabels(q_vol["period"], rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
# Annotate last 4 bars
for bar in bars[-4:]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{bar.get_height():.1f}B", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color=PHONEPE_PURPLE)
save(fig, "01_quarterly_volume.png")


# ── Plot 2: QoQ Growth Rate ──────────────────────────────────────────────────
print("[2/8] QoQ growth rate...")
q_vol["qoq"] = q_vol["txns_bn"].pct_change() * 100
q_vol_plot = q_vol.dropna(subset=["qoq"])

fig, ax = plt.subplots(figsize=(14, 4))
colors = [PHONEPE_GREEN if v >= 0 else ACCENT for v in q_vol_plot["qoq"]]
ax.bar(q_vol_plot["period"], q_vol_plot["qoq"], color=colors, alpha=0.85, width=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Quarter-over-Quarter Transaction Growth Rate (%)")
ax.set_ylabel("QoQ Growth (%)")
ax.set_xticks(range(len(q_vol_plot)))
ax.set_xticklabels(q_vol_plot["period"], rotation=45, ha="right", fontsize=8)
save(fig, "02_qoq_growth.png")


# ── Plot 3: Transaction Type Mix — stacked area chart ───────────────────────
print("[3/8] Transaction type mix shift...")
types = ["Merchant payments", "Peer-to-peer payments",
         "Recharge & bill payments", "Financial Services", "Others"]
colors_type = [PHONEPE_PURPLE, PHONEPE_GREEN, "#f7941d", ACCENT, "#888888"]

pivot = (df_national.groupby(["year","quarter","transaction_type"])["txn_count"]
         .sum().reset_index())
pivot["period"] = pivot["year"].astype(str) + "-Q" + pivot["quarter"].astype(str)
pivot_wide = pivot.pivot_table(index="period", columns="transaction_type",
                                values="txn_count", fill_value=0)
pivot_wide = pivot_wide.div(pivot_wide.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 5))
bottom = np.zeros(len(pivot_wide))
for i, (t, c) in enumerate(zip(types, colors_type)):
    if t in pivot_wide.columns:
        vals = pivot_wide[t].values
        ax.fill_between(range(len(pivot_wide)), bottom, bottom + vals,
                        alpha=0.85, color=c, label=t)
        bottom += vals

ax.set_title("Transaction Type Mix Shift (2018–2024) — PhonePe Pulse")
ax.set_ylabel("Share of Transactions (%)")
ax.set_xticks(range(len(pivot_wide)))
ax.set_xticklabels(pivot_wide.index, rotation=45, ha="right", fontsize=8)
ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
ax.set_ylim(0, 100)
save(fig, "03_txn_type_mix.png")


# ── Plot 4: Top 10 States — horizontal bar ───────────────────────────────────
print("[4/8] Top 10 states...")
top_states = (df_states[df_states["year"] == 2024]
              .groupby("state")["txn_count"].sum()
              .nlargest(10).reset_index())
top_states["txns_bn"] = top_states["txn_count"] / 1e9
top_states["state_clean"] = top_states["state"].str.replace("-", " ").str.title()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_states["state_clean"], top_states["txns_bn"],
               color=PHONEPE_PURPLE, alpha=0.85)
ax.set_title("Top 10 States by Total UPI Transactions (2024)")
ax.set_xlabel("Transactions (Billions)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}B"))
for bar in bars:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}B", va="center", fontsize=9)
ax.invert_yaxis()
save(fig, "04_top_states.png")


# ── Plot 5: User Growth — line chart ─────────────────────────────────────────
print("[5/8] User growth trajectory...")
user_nat = (df_users[df_users["state"] == "india"]
            .groupby("year")["registered_users"].max().reset_index())
user_nat["users_mn"] = user_nat["registered_users"] / 1e6

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(user_nat["year"], user_nat["users_mn"], marker="o", linewidth=2.5,
        color=PHONEPE_PURPLE, markersize=8)
ax.fill_between(user_nat["year"], user_nat["users_mn"], alpha=0.15, color=PHONEPE_PURPLE)
for _, row in user_nat.iterrows():
    ax.annotate(f"{row['users_mn']:.0f}M", (row["year"], row["users_mn"]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, fontweight="bold")
ax.set_title("PhonePe Registered Users Growth (2018–2024)")
ax.set_ylabel("Registered Users (Millions)")
ax.set_xlabel("Year")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
save(fig, "05_user_growth.png")


# ── Plot 6: Merchant vs P2P split — grouped bar by year ─────────────────────
print("[6/8] Merchant vs P2P split...")
mp = df_national[df_national["transaction_type"].isin(
    ["Merchant payments","Peer-to-peer payments"])].copy()
mp_pivot = mp.pivot_table(index="year", columns="transaction_type",
                           values="txn_count", aggfunc="sum") / 1e9

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(mp_pivot))
w = 0.38
ax.bar(x - w/2, mp_pivot["Merchant payments"],  width=w, color=PHONEPE_PURPLE,
       label="Merchant", alpha=0.9)
ax.bar(x + w/2, mp_pivot["Peer-to-peer payments"], width=w, color=PHONEPE_GREEN,
       label="P2P", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(mp_pivot.index)
ax.set_title("Merchant Payments vs P2P — Transaction Volume (Billions)")
ax.set_ylabel("Transactions (Billions)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}B"))
save(fig, "06_merchant_vs_p2p.png")


# ── Plot 7: 4-Quarter Moving Average — smoothed trend ────────────────────────
print("[7/8] Rolling moving average...")
q_all = (df_national.groupby(["year","quarter"])[["txn_count","txn_amount"]]
         .sum().reset_index().sort_values(["year","quarter"]))
q_all["period"]   = q_all["year"].astype(str) + "-Q" + q_all["quarter"].astype(str)
q_all["txns_bn"]  = q_all["txn_count"] / 1e9
q_all["ma_4q"]    = q_all["txns_bn"].rolling(4).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(q_all)), q_all["txns_bn"], color=PHONEPE_PURPLE,
       alpha=0.4, label="Actual", width=0.7)
ax.plot(range(len(q_all)), q_all["ma_4q"], color=ACCENT, linewidth=2.5,
        marker="o", markersize=4, label="4Q Moving Average")
ax.set_xticks(range(len(q_all)))
ax.set_xticklabels(q_all["period"], rotation=45, ha="right", fontsize=8)
ax.set_title("UPI Transaction Trend — Actual vs 4-Quarter Moving Average")
ax.set_ylabel("Transactions (Billions)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}B"))
save(fig, "07_moving_average.png")


# ── Plot 8: State heatmap — Avg transaction value Z-score ────────────────────
print("[8/8] State Z-score heatmap...")
state_2024 = (df_states[df_states["year"] == 2024]
              .groupby("state").agg(
                  total_count  = ("txn_count",  "sum"),
                  total_amount = ("txn_amount", "sum")
              ).reset_index())
state_2024["avg_txn"] = state_2024["total_amount"] / state_2024["total_count"]
state_2024["z_score"] = ((state_2024["avg_txn"] - state_2024["avg_txn"].mean())
                         / state_2024["avg_txn"].std())
state_2024["state_clean"] = state_2024["state"].str.replace("-", " ").str.title()
state_2024 = state_2024.sort_values("z_score")

fig, ax = plt.subplots(figsize=(9, 11))
colors_z = [ACCENT if z > 1.5 else (PHONEPE_GREEN if z < -1.5 else PHONEPE_PURPLE)
            for z in state_2024["z_score"]]
ax.barh(state_2024["state_clean"], state_2024["z_score"], color=colors_z, alpha=0.85)
ax.axvline(0,    color="black",       linewidth=0.8)
ax.axvline(1.5,  color=ACCENT,        linewidth=1,  linestyle="--", alpha=0.7)
ax.axvline(-1.5, color=PHONEPE_GREEN, linewidth=1,  linestyle="--", alpha=0.7)
ax.set_title("Avg Transaction Value Z-Score by State (2024)\nOutliers flagged at ±1.5σ")
ax.set_xlabel("Z-Score")
patches = [
    mpatches.Patch(color=ACCENT,        label="High-value outlier (z > 1.5)"),
    mpatches.Patch(color=PHONEPE_PURPLE,label="Normal range"),
    mpatches.Patch(color=PHONEPE_GREEN, label="Low-value outlier (z < -1.5)"),
]
ax.legend(handles=patches, loc="lower right", fontsize=9)
save(fig, "08_state_zscore.png")


# ── Excel Export — 5-sheet formatted workbook ────────────────────────────────
print("\n[Excel] Building workbook...")

MAX_YEAR = int(df_states["year"].max())


def write_sheet(writer, df, sheet, title, formats, widths, index=False):
    """Write a frame with a banner title ABOVE the header row.

    The header is offset by startrow=2 so the title never overwrites a column
    name — the original version wrote the title into A1, silently destroying
    the first column's header.
    """
    df.to_excel(writer, sheet_name=sheet, index=index, startrow=2)
    ws = writer.sheets[sheet]
    ws.write(0, 0, title, formats["title"])

    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    for i, col in enumerate(cols):
        ws.write(2, i, col, formats["hdr"])
    for span, fmt in widths:
        ws.set_column(span, None, fmt)
    ws.freeze_panes(3, 1 if index else 0)
    return ws


with pd.ExcelWriter(XLSX_PATH, engine="xlsxwriter") as writer:
    wb = writer.book

    F = {
        "hdr":   wb.add_format({"bold": True, "bg_color": "#5f259f",
                                "font_color": "white", "border": 1,
                                "align": "center", "valign": "vcenter",
                                "text_wrap": True}),
        "num":   wb.add_format({"num_format": "#,##0", "border": 1}),
        "pct":   wb.add_format({"num_format": '0.0"%"', "border": 1}),
        "amt":   wb.add_format({"num_format": "₹#,##0", "border": 1}),
        "dec":   wb.add_format({"num_format": "#,##0.00", "border": 1}),
        "txt":   wb.add_format({"border": 1}),
        "title": wb.add_format({"bold": True, "font_size": 13,
                                "font_color": "#5f259f"}),
    }
    red_fmt   = wb.add_format({"bg_color": "#ffe0e0", "font_color": "#9c0006"})
    green_fmt = wb.add_format({"bg_color": "#e0ffe0", "font_color": "#006100"})

    # ── Sheet 1: Quarterly Summary ───────────────────────────────────────────
    q_sum = q_all.copy()
    q_sum["txn_amount_bn"] = (q_sum["txn_amount"] / 1e9).round(2)
    q_sum["txns_mn"]       = (q_sum["txn_count"]  / 1e6).round(1)
    q_sum["qoq_pct"]       = (q_sum["txn_count"].pct_change() * 100).round(1)
    q_sum["avg_ticket"]    = (q_sum["txn_amount"] / q_sum["txn_count"]).round(0)
    sheet1 = q_sum[["period", "txns_mn", "txn_amount_bn", "qoq_pct", "avg_ticket"]].rename(
        columns={"period": "Quarter", "txns_mn": "Transactions (M)",
                 "txn_amount_bn": "Amount (₹B)", "qoq_pct": "QoQ Growth %",
                 "avg_ticket": "Avg Ticket (₹)"})
    ws = write_sheet(
        writer, sheet1, "Quarterly Summary",
        "PhonePe UPI — Quarterly Transaction Summary", F,
        [("A:A", None), ("B:C", F["dec"]), ("D:D", F["pct"]), ("E:E", F["amt"])])
    ws.set_column("A:A", 14); ws.set_column("B:E", 18)
    n1 = len(sheet1)
    ws.conditional_format(f"D4:D{n1+3}", {"type": "cell", "criteria": "<",
                                          "value": 0, "format": red_fmt})
    ws.conditional_format(f"B4:B{n1+3}", {"type": "data_bar",
                                          "bar_color": "#8247d4"})

    # ── Sheet 2: State Pivot (latest year, no longer hardcoded) ─────────────
    state_pivot = (df_states[df_states["year"] == MAX_YEAR]
                   .groupby(["state", "transaction_type"])["txn_count"]
                   .sum().reset_index()
                   .pivot_table(index="state", columns="transaction_type",
                                values="txn_count", aggfunc="sum", fill_value=0))
    state_pivot["Total"] = state_pivot.sum(axis=1)
    state_pivot["Market Share %"] = (state_pivot["Total"] /
                                      state_pivot["Total"].sum() * 100).round(2)
    state_pivot = state_pivot.sort_values("Total", ascending=False)
    ws2 = write_sheet(
        writer, state_pivot, f"State Pivot {MAX_YEAR}",
        f"Transaction volume by state and type — {MAX_YEAR}", F,
        [("B:H", F["num"])], index=True)
    ws2.set_column("A:A", 38); ws2.set_column("B:H", 18)
    ws2.conditional_format(f"I4:I{len(state_pivot)+3}",
                           {"type": "data_bar", "bar_color": "#5f259f"})

    # ── Sheet 3: Transaction Type Trend ─────────────────────────────────────
    type_trend = (df_national.groupby(["year", "transaction_type"])["txn_count"]
                  .sum().reset_index()
                  .pivot_table(index="year", columns="transaction_type",
                               values="txn_count", aggfunc="sum", fill_value=0))
    ws3 = write_sheet(
        writer, type_trend, "Type Trend by Year",
        "Transaction mix shift by year (counts)", F,
        [("B:G", F["num"])], index=True)
    ws3.set_column("A:G", 22)

    # ── Sheet 4: Type Mix % — the actual story, as shares ────────────────────
    type_mix = (type_trend.div(type_trend.sum(axis=1), axis=0) * 100).round(1)
    ws4 = write_sheet(
        writer, type_mix, "Type Mix %",
        "Transaction mix shift by year (% of volume)", F,
        [("B:G", F["pct"])], index=True)
    ws4.set_column("A:G", 22)
    ws4.conditional_format(f"B4:G{len(type_mix)+3}",
                           {"type": "3_color_scale",
                            "min_color": "#ffffff", "mid_color": "#c9aee6",
                            "max_color": "#5f259f"})

    # ── Sheet 5: Merchant-adjusted anomalies ────────────────────────────────
    _latest = df_states[df_states["year"] == MAX_YEAR]
    _base = (_latest.groupby("state")
             .agg(txn_count=("txn_count", "sum"), txn_amount=("txn_amount", "sum"))
             .reset_index())
    _merch = (_latest[_latest["transaction_type"] == "Merchant payments"]
              .groupby("state")["txn_count"].sum().rename("merchant_count").reset_index())
    _base = _base.merge(_merch, on="state", how="left")
    _base["merchant_count"] = _base["merchant_count"].fillna(0)

    _res = merchant_adjusted_outliers(_base)
    _res["State"] = _res["state"].map(normalise_state)
    anomaly_df = (_res.sort_values("residual_z", ascending=False)[
        ["State", "merchant_pct", "avg_ticket", "expected_ticket",
         "residual", "residual_z", "flag"]]
        .round({"merchant_pct": 1, "avg_ticket": 0, "expected_ticket": 0,
                "residual": 0, "residual_z": 2})
        .rename(columns={"merchant_pct": "Merchant %",
                         "avg_ticket": "Actual Ticket (₹)",
                         "expected_ticket": "Expected Ticket (₹)",
                         "residual": "Residual (₹)",
                         "residual_z": "Residual Z",
                         "flag": "Flag"}))
    ws5 = write_sheet(
        writer, anomaly_df, "Anomaly Flags",
        f"Avg ticket vs merchant mix — residual outliers ({MAX_YEAR}), "
        f"r = {merchant_ticket_correlation(_base):.2f}", F,
        [("B:B", F["pct"]), ("C:E", F["amt"]), ("F:F", F["dec"])])
    ws5.set_column("A:A", 38); ws5.set_column("B:G", 20)
    nr = len(anomaly_df) + 3
    ws5.conditional_format(f"G4:G{nr}", {"type": "text", "criteria": "containing",
                                         "value": "ABOVE", "format": red_fmt})
    ws5.conditional_format(f"G4:G{nr}", {"type": "text", "criteria": "containing",
                                         "value": "BELOW", "format": green_fmt})

print(f"  ✅ Excel saved: {XLSX_PATH}")
print("\n✅ EDA complete — 8 plots + Excel workbook generated.")
conn.close()
