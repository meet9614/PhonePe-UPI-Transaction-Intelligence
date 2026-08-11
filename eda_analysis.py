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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
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

DB_PATH   = Path("data/phonepe_pulse.db")
OUT_DIR   = Path("assets/eda_plots")
XLSX_PATH = Path("data/phonepe_summary.xlsx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


# ── Excel Export — 4-sheet formatted workbook ────────────────────────────────
print("\n[Excel] Building workbook...")

with pd.ExcelWriter(XLSX_PATH, engine="xlsxwriter") as writer:
    wb = writer.book

    # Formats
    hdr_fmt   = wb.add_format({"bold":True, "bg_color":"#5f259f",
                                "font_color":"white", "border":1})
    num_fmt   = wb.add_format({"num_format":"#,##0",   "border":1})
    pct_fmt   = wb.add_format({"num_format":"0.0\"%\"", "border":1})
    amt_fmt   = wb.add_format({"num_format":"#,##0.00", "border":1})
    red_fmt   = wb.add_format({"bg_color":"#ffe0e0", "border":1})
    green_fmt = wb.add_format({"bg_color":"#e0ffe0", "border":1})
    title_fmt = wb.add_format({"bold":True, "font_size":13,
                                "font_color":"#5f259f"})

    # ── Sheet 1: Quarterly Summary ───────────────────────────────────────────
    q_sum = q_all.copy()
    q_sum["txn_amount_bn"] = (q_sum["txn_amount"] / 1e9).round(2)
    q_sum["txns_mn"]       = (q_sum["txn_count"]  / 1e6).round(1)
    q_sum["qoq_pct"]       = q_sum["txns_mn"].pct_change().mul(100).round(1)
    sheet1 = q_sum[["period","txns_mn","txn_amount_bn","qoq_pct"]].rename(columns={
        "period":"Quarter","txns_mn":"Transactions (M)",
        "txn_amount_bn":"Amount (₹B)","qoq_pct":"QoQ Growth %"})
    sheet1.to_excel(writer, sheet_name="Quarterly Summary", index=False)
    ws = writer.sheets["Quarterly Summary"]
    ws.write("A1", "PhonePe UPI — Quarterly Transaction Summary", title_fmt)
    ws.set_row(0, 20, title_fmt)
    ws.set_column("A:A", 14); ws.set_column("B:D", 18)

    # ── Sheet 2: State Pivot (latest year) ──────────────────────────────────
    state_pivot = (df_states[df_states["year"] == 2024]
                   .groupby(["state","transaction_type"])["txn_count"]
                   .sum().reset_index()
                   .pivot_table(index="state", columns="transaction_type",
                                values="txn_count", aggfunc="sum", fill_value=0))
    state_pivot["Total"] = state_pivot.sum(axis=1)
    state_pivot["Market Share %"] = (state_pivot["Total"] /
                                      state_pivot["Total"].sum() * 100).round(2)
    state_pivot = state_pivot.sort_values("Total", ascending=False)
    state_pivot.to_excel(writer, sheet_name="State Pivot 2024")
    ws2 = writer.sheets["State Pivot 2024"]
    ws2.set_column("A:A", 38)
    ws2.set_column("B:H", 18)

    # ── Sheet 3: Transaction Type Trend ─────────────────────────────────────
    type_trend = (df_national.groupby(["year","transaction_type"])["txn_count"]
                  .sum().reset_index()
                  .pivot_table(index="year", columns="transaction_type",
                               values="txn_count", aggfunc="sum", fill_value=0))
    type_trend.to_excel(writer, sheet_name="Type Trend by Year")
    ws3 = writer.sheets["Type Trend by Year"]
    ws3.set_column("A:G", 22)

    # ── Sheet 4: Anomaly Flags ───────────────────────────────────────────────
    anomaly_df = state_2024[["state_clean","avg_txn","z_score"]].copy()
    anomaly_df["avg_txn"] = anomaly_df["avg_txn"].round(0)
    anomaly_df["z_score"] = anomaly_df["z_score"].round(2)
    anomaly_df["Flag"] = anomaly_df["z_score"].apply(
        lambda z: "HIGH_VALUE_OUTLIER" if z > 1.5
                  else ("LOW_VALUE_OUTLIER" if z < -1.5 else "NORMAL"))
    anomaly_df = anomaly_df.rename(columns={
        "state_clean":"State","avg_txn":"Avg Txn Value (₹)","z_score":"Z-Score"})
    anomaly_df.sort_values("Z-Score", ascending=False).to_excel(
        writer, sheet_name="Anomaly Flags", index=False)
    ws4 = writer.sheets["Anomaly Flags"]
    ws4.set_column("A:A", 38); ws4.set_column("B:D", 22)

print(f"  ✅ Excel saved: {XLSX_PATH}")
print("\n✅ EDA complete — 8 plots + Excel workbook generated.")
conn.close()
