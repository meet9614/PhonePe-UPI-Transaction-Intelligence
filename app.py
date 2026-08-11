"""
PhonePe UPI Transaction Intelligence — Streamlit Dashboard
============================================================
Run:  streamlit run app.py
"""

import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhonePe UPI Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ─────────────────────────────────────────────────────────────
PP_PURPLE = "#5f259f"
PP_LIGHT  = "#8247d4"
PP_GREEN  = "#00b9f5"
PP_ORANGE = "#f7941d"
PP_RED    = "#e8453c"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f5f5f5; }
[data-testid="stSidebar"]          { background: #1a0533; }
[data-testid="stSidebar"] * { color: #e0d0f5 !important; }
.kpi-card {
    background: white; border-radius: 12px;
    padding: 18px 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 5px solid #5f259f;
}
.kpi-value { font-size: 28px; font-weight: 700; color: #5f259f; }
.kpi-label { font-size: 13px; color: #666; margin-bottom: 4px; }
.kpi-delta { font-size: 12px; color: #2e7d32; margin-top: 4px; }
.kpi-delta-neg { font-size: 12px; color: #c62828; margin-top: 4px; }
.section-header {
    font-size: 17px; font-weight: 700;
    color: #5f259f; margin: 18px 0 8px 0;
    border-bottom: 2px solid #5f259f; padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Data layer ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "data" / "phonepe_pulse.db"


@st.cache_data(show_spinner="Loading PhonePe Pulse data…")
def load_data(db_path: str, _mtime: float):
    """Read the three aggregate tables. `_mtime` busts the cache if the DB changes."""
    with sqlite3.connect(db_path) as conn:
        txn   = pd.read_sql("SELECT * FROM agg_transactions", conn)
        users = pd.read_sql("SELECT * FROM agg_users",        conn)
        top   = pd.read_sql("SELECT * FROM top_entities",     conn)
    return txn, users, top


if not DB_PATH.exists():
    st.error(
        f"Database not found at `{DB_PATH}`.\n\n"
        "Run `python data_ingestion.py` from the project folder first."
    )
    st.stop()

df_txn, df_users, df_top = load_data(str(DB_PATH), DB_PATH.stat().st_mtime)
df_nat    = df_txn[df_txn["state"] == "india"].copy()
df_states = df_txn[df_txn["state"] != "india"].copy()
YEARS     = sorted(df_txn["year"].unique())
MAX_YEAR  = max(YEARS)


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 PhonePe UPI\n### Transaction Intelligence")
    st.markdown("---")
    st.markdown("**Filters**")

    year_range = st.slider(
        "Year range", min_value=int(min(YEARS)),
        max_value=int(MAX_YEAR), value=(2020, int(MAX_YEAR))
    )
    all_types = df_txn["transaction_type"].unique().tolist()
    selected_types = st.multiselect(
        "Transaction types", options=all_types, default=all_types
    )
    if not selected_types:
        st.warning("No transaction types selected — showing all.")
        selected_types = all_types
    top_n_states = st.slider("Top N states", 5, 20, 10)

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- Years: **{min(YEARS)} – {MAX_YEAR}**")
    st.markdown(f"- States: **{df_states['state'].nunique()}**")
    st.markdown(f"- Transaction rows: **{len(df_txn):,}**")
    st.markdown("---")
    st.markdown("*Built on PhonePe Pulse*  \n*[github.com/PhonePe/pulse](https://github.com/PhonePe/pulse)*")


# ── Apply filters ─────────────────────────────────────────────────────────────
mask_nat = (
    (df_nat["year"] >= year_range[0]) &
    (df_nat["year"] <= year_range[1]) &
    (df_nat["transaction_type"].isin(selected_types))
)
mask_st = (
    (df_states["year"] >= year_range[0]) &
    (df_states["year"] <= year_range[1]) &
    (df_states["transaction_type"].isin(selected_types))
)
filt_nat    = df_nat[mask_nat]
filt_states = df_states[mask_st]

if filt_nat.empty:
    st.warning("No data matches the current filters. Widen the year range in the sidebar.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#5f259f;margin-bottom:0'>📲 PhonePe UPI Transaction Intelligence</h1>"
    "<p style='color:#666;margin-top:4px'>Built on PhonePe Pulse open data &nbsp;|&nbsp; "
    "2018–2024 &nbsp;|&nbsp; 36 states &nbsp;|&nbsp; 100B+ transactions</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
q_vol = (filt_nat.groupby(["year","quarter"])
         .agg(txns=("txn_count","sum"), amt=("txn_amount","sum"))
         .reset_index().sort_values(["year","quarter"]))

total_txns  = filt_nat["txn_count"].sum()
total_amt   = filt_nat["txn_amount"].sum()
latest_qoq  = float((q_vol["txns"].pct_change() * 100).iloc[-1]) if len(q_vol) > 1 else 0.0
peak_users  = df_users[(df_users["state"]=="india") &
                        (df_users["year"]==MAX_YEAR)]["registered_users"].max()
peak_users  = 0 if pd.isna(peak_users) else peak_users

k1, k2, k3, k4 = st.columns(4)

def kpi(col, label, value, delta=None, delta_label="vs prev quarter"):
    sign   = "+" if (delta is not None and delta >= 0) else ""
    dclass = "kpi-delta" if (delta is None or delta >= 0) else "kpi-delta-neg"
    delta_html = (f"<div class='{dclass}'>{sign}{delta:.1f}% {delta_label}</div>"
                  if delta is not None else "")
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True
    )

with k1: kpi(k1, "Total Transactions", f"{total_txns/1e9:.1f}B", latest_qoq)
with k2: kpi(k2, "Total Value Processed", f"₹{total_amt/1e12:.1f}T")
with k3: kpi(k3, "Registered Users (Peak)", f"{peak_users/1e6:.0f}M")
with k4:
    merch = filt_nat[filt_nat["transaction_type"]=="Merchant payments"]["txn_count"].sum()
    merch_pct = 100 * merch / total_txns if total_txns else 0
    kpi(k4, "Merchant Payment Share", f"{merch_pct:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 1: Quarterly Volume  |  QoQ Growth
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📈 Transaction Volume & Growth Trend</div>",
            unsafe_allow_html=True)
col_l, col_r = st.columns([3, 2])

with col_l:
    q_vol["period"] = q_vol["year"].astype(str) + "-Q" + q_vol["quarter"].astype(str)
    q_vol["txns_bn"] = q_vol["txns"] / 1e9
    q_vol["ma4"] = q_vol["txns_bn"].rolling(4).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=q_vol["period"], y=q_vol["txns_bn"],
        name="Quarterly Volume", marker_color=PP_PURPLE, opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=q_vol["period"], y=q_vol["ma4"],
        name="4Q Moving Avg", line=dict(color=PP_RED, width=2.5),
        mode="lines+markers", marker_size=5
    ))
    fig.update_layout(
        title="Quarterly Transaction Volume (Billions) + 4Q MA",
        xaxis_tickangle=-45, legend=dict(orientation="h", y=1.1),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=40), height=340,
        yaxis_title="Transactions (B)"
    )
    st.plotly_chart(fig, width="stretch")

with col_r:
    q_vol["qoq"] = q_vol["txns_bn"].pct_change() * 100
    q_last = q_vol.dropna(subset=["qoq"]).tail(12)
    colors_q = [PP_GREEN if v >= 0 else PP_RED for v in q_last["qoq"]]
    fig2 = go.Figure(go.Bar(
        x=q_last["period"], y=q_last["qoq"],
        marker_color=colors_q, opacity=0.85
    ))
    fig2.update_layout(
        title="QoQ Growth Rate (%)",
        xaxis_tickangle=-45, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=40), height=340,
        yaxis_title="QoQ %"
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    st.plotly_chart(fig2, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 2: State Market Share  |  Transaction Type Mix
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🗺️ State & Category Analysis</div>",
            unsafe_allow_html=True)
col_l2, col_r2 = st.columns([2, 3])

with col_l2:
    state_agg = (filt_states.groupby("state")["txn_count"]
                 .sum().nlargest(top_n_states).reset_index())
    state_agg["state_clean"] = (state_agg["state"].str.replace("-", " ")
                                 .str.title())
    state_agg["txns_bn"] = state_agg["txn_count"] / 1e9
    state_agg["share_pct"] = (state_agg["txn_count"] /
                               state_agg["txn_count"].sum() * 100).round(1)

    fig3 = go.Figure(go.Bar(
        x=state_agg["txns_bn"], y=state_agg["state_clean"],
        orientation="h", marker_color=PP_PURPLE, opacity=0.85,
        text=state_agg["share_pct"].apply(lambda x: f"{x}%"),
        textposition="outside"
    ))
    fig3.update_layout(
        title=f"Top {top_n_states} States by Transaction Volume",
        xaxis_title="Transactions (Billions)", yaxis_autorange="reversed",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, l=10, r=60), height=400
    )
    st.plotly_chart(fig3, width="stretch")

with col_r2:
    type_trend = (filt_nat.groupby(["year","transaction_type"])["txn_count"]
                  .sum().reset_index())
    type_trend["txns_bn"] = type_trend["txn_count"] / 1e9

    fig4 = px.area(
        type_trend, x="year", y="txns_bn", color="transaction_type",
        color_discrete_sequence=[PP_PURPLE, PP_GREEN, PP_ORANGE, PP_RED, "#aaaaaa"],
        labels={"txns_bn": "Transactions (B)", "year": "Year",
                "transaction_type": "Type"},
        title="Transaction Type Mix Shift (Stacked Area)"
    )
    fig4.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.25, font_size=11),
        margin=dict(t=50, b=80), height=400
    )
    st.plotly_chart(fig4, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 3: User Growth  |  Anomaly Detection Table
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>👥 User Adoption & Anomaly Detection</div>",
            unsafe_allow_html=True)
col_l3, col_r3 = st.columns([2, 3])

with col_l3:
    user_yr = (df_users[(df_users["state"] == "india") &
                         (df_users["year"] >= year_range[0]) &
                         (df_users["year"] <= year_range[1])]
               .groupby("year")["registered_users"].max().reset_index())
    user_yr["users_mn"] = user_yr["registered_users"] / 1e6
    user_yr["yoy"] = user_yr["users_mn"].pct_change() * 100

    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Scatter(
        x=user_yr["year"], y=user_yr["users_mn"],
        name="Registered Users (M)", fill="tozeroy",
        line=dict(color=PP_PURPLE, width=2.5),
        fillcolor=f"rgba(95,37,159,0.15)"
    ), secondary_y=False)
    fig5.add_trace(go.Bar(
        x=user_yr["year"], y=user_yr["yoy"],
        name="YoY Growth %", marker_color=PP_GREEN,
        opacity=0.6, width=0.4
    ), secondary_y=True)
    fig5.update_layout(
        title="Registered User Growth + YoY %",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=55, b=30), height=340
    )
    fig5.update_yaxes(title_text="Users (Millions)", secondary_y=False)
    fig5.update_yaxes(title_text="YoY Growth %",     secondary_y=True)
    st.plotly_chart(fig5, width="stretch")

with col_r3:
    # Z-score anomaly table
    st_2024 = (df_states[df_states["year"] == MAX_YEAR]
               .groupby("state").agg(
                   total_count  =("txn_count",  "sum"),
                   total_amount =("txn_amount", "sum")
               ).reset_index())
    st_2024["avg_txn_value"] = (st_2024["total_amount"] /
                                  st_2024["total_count"]).round(0)
    mean_ = st_2024["avg_txn_value"].mean()
    std_  = st_2024["avg_txn_value"].std()
    st_2024["z_score"] = ((st_2024["avg_txn_value"] - mean_) / std_).round(2)
    st_2024["flag"] = st_2024["z_score"].apply(
        lambda z: "🔴 HIGH VALUE" if z >  1.5
             else ("🔵 LOW VALUE"  if z < -1.5 else "✅ NORMAL"))
    st_2024["state"] = st_2024["state"].str.replace("-", " ").str.title()
    st_2024["txns_mn"] = (st_2024["total_count"] / 1e6).round(1)

    display = (st_2024.rename(columns={
        "state":         "State",
        "txns_mn":       "Txns (M)",
        "avg_txn_value": "Avg ₹/Txn",
        "z_score":       "Z-Score",
        "flag":          "Anomaly Flag"
    })[["State","Txns (M)","Avg ₹/Txn","Z-Score","Anomaly Flag"]]
    .sort_values("Z-Score", ascending=False))

    st.markdown(f"##### Avg Transaction Value Z-Score by State ({MAX_YEAR})")
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=330,
        column_config={
            "Txns (M)":  st.column_config.NumberColumn(format="%.1f M"),
            "Avg ₹/Txn": st.column_config.NumberColumn(format="₹%.0f"),
            "Z-Score":   st.column_config.NumberColumn(format="%.2f"),
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 4: Merchant vs P2P deep-dive
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🏪 Merchant Payment Penetration</div>",
            unsafe_allow_html=True)

mp_data = filt_states[filt_states["transaction_type"].isin(
    ["Merchant payments", "Peer-to-peer payments"])].copy()
mp_pivot = (mp_data.groupby(["state","transaction_type"])["txn_count"]
            .sum().reset_index()
            .pivot_table(index="state", columns="transaction_type",
                         values="txn_count", fill_value=0).reset_index())

if "Merchant payments" in mp_pivot.columns and "Peer-to-peer payments" in mp_pivot.columns:
    mp_pivot["total"] = mp_pivot["Merchant payments"] + mp_pivot["Peer-to-peer payments"]
    mp_pivot["merchant_pct"] = (mp_pivot["Merchant payments"] /
                                  mp_pivot["total"] * 100).round(1)
    mp_pivot = mp_pivot.nlargest(top_n_states, "total")
    mp_pivot["state_clean"] = (mp_pivot["state"].str.replace("-", " ").str.title())
    mp_pivot = mp_pivot.sort_values("merchant_pct", ascending=True)

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        name="Merchant Payments",
        y=mp_pivot["state_clean"],
        x=mp_pivot["Merchant payments"] / 1e6,
        orientation="h", marker_color=PP_PURPLE, opacity=0.85
    ))
    fig6.add_trace(go.Bar(
        name="P2P Payments",
        y=mp_pivot["state_clean"],
        x=mp_pivot["Peer-to-peer payments"] / 1e6,
        orientation="h", marker_color=PP_GREEN, opacity=0.85
    ))
    fig6.update_layout(
        barmode="stack",
        title=f"Merchant vs P2P Volume — Top {top_n_states} States (Millions of transactions)",
        xaxis_title="Transactions (Millions)",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=55, b=30), height=380
    )
    st.plotly_chart(fig6, width="stretch")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:12px'>"
    "Data: PhonePe Pulse (github.com/PhonePe/pulse) &nbsp;|&nbsp; "
    "Built by Meet Kumar Sarkar, NIT Patna &nbsp;|&nbsp; "
    "Stack: Python · SQL · Streamlit · Plotly"
    "</div>",
    unsafe_allow_html=True
)
