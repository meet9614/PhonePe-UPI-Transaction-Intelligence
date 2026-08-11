"""
PhonePe UPI Transaction Intelligence — Streamlit Dashboard
============================================================
Run:  streamlit run app.py

Requires Streamlit >= 1.49 (uses the width="stretch" layout API).
"""

import json
import sqlite3
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analytics as A

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
.takeaway {
    background: #f3ecfa; border-left: 4px solid #8247d4;
    padding: 10px 14px; margin: 4px 0 18px 0;
    font-size: 13.5px; color: #3d2159; border-radius: 0 6px 6px 0;
}
</style>
""", unsafe_allow_html=True)


def takeaway(text: str) -> None:
    """Render the 'so what' line under a chart."""
    st.markdown(f"<div class='takeaway'>{text}</div>", unsafe_allow_html=True)


# ── Data layer ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "data" / "phonepe_pulse.db"


@st.cache_data(show_spinner="Loading PhonePe Pulse data…")
def load_data(db_path: str, _mtime: float):
    """Read the aggregate tables. `_mtime` busts the cache if the DB changes."""
    with sqlite3.connect(db_path) as conn:
        txn     = pd.read_sql("SELECT * FROM agg_transactions",  conn)
        users   = pd.read_sql("SELECT * FROM agg_users",         conn)
        top     = pd.read_sql("SELECT * FROM top_entities",      conn)
        devices = pd.read_sql("SELECT * FROM agg_user_devices",  conn)
    return txn, users, top, devices


@st.cache_data(show_spinner="Fetching India map…", ttl=86_400)
def load_geojson(url: str):
    """India state boundaries. Returns None if unreachable so the map can
    degrade to a bar chart rather than taking the whole page down."""
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


if not DB_PATH.exists():
    st.error(
        f"Database not found at `{DB_PATH}`.\n\n"
        "Run `python data_ingestion.py` from the project folder first."
    )
    st.stop()

df_txn, df_users, df_top, df_dev = load_data(str(DB_PATH), DB_PATH.stat().st_mtime)
df_pop    = A.load_population()
df_nat    = df_txn[df_txn["state"] == "india"].copy()
df_states = df_txn[df_txn["state"] != "india"].copy()
YEARS     = sorted(df_txn["year"].unique())
MAX_YEAR  = int(max(YEARS))


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 PhonePe UPI\n### Transaction Intelligence")
    st.markdown("---")
    st.markdown("**Filters**")

    year_range = st.slider(
        "Year range", min_value=int(min(YEARS)),
        max_value=MAX_YEAR, value=(2020, MAX_YEAR)
    )
    all_types = df_txn["transaction_type"].unique().tolist()
    selected_types = st.multiselect(
        "Transaction types", options=all_types, default=all_types
    )
    if not selected_types:
        st.warning("No transaction types selected — showing all.")
        selected_types = all_types
    top_n_states = st.slider("Top N states", 5, 20, 10)
    per_capita = st.toggle(
        "Normalise per capita", value=False,
        help="Divide by Census 2011 population. Turns 'biggest states' into "
             "'states where UPI actually penetrated'."
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- Years: **{min(YEARS)} – {MAX_YEAR}**")
    st.markdown(f"- States: **{df_states['state'].nunique()}**")
    st.markdown(f"- Districts: **{df_top[df_top.entity_level=='district'].entity_name.nunique()}**")
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
    "2018–2024 &nbsp;|&nbsp; 36 states &nbsp;|&nbsp; 235B transactions &nbsp;|&nbsp; ₹345T</p>",
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
avg_ticket  = total_amt / total_txns if total_txns else 0

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
with k4: kpi(k4, "Average Ticket Size", f"₹{avg_ticket:,.0f}")

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

takeaway(
    "<b>Growth is decelerating on schedule, not stalling.</b> Average QoQ growth fell "
    "from 47.9% (2018) to 10.4% (2024) — the classic S-curve shape, not a demand problem. "
    "2020-Q2 is the only negative quarter in 28 (−10.7%, the COVID lockdown), and it "
    "rebounded +40% the following quarter."
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 2: Choropleth map
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🗺️ Geographic Penetration</div>",
            unsafe_allow_html=True)

map_agg = (filt_states.groupby("state")["txn_count"].sum().reset_index())
map_agg = map_agg.merge(df_pop, on="state", how="left")
map_agg["per_capita"] = map_agg["txn_count"] / map_agg["population_2011"]
map_agg["State"] = map_agg["state"].map(A.normalise_state)
map_agg["geo_key"] = map_agg["state"].map(A.geojson_key)

metric_col   = "per_capita" if per_capita else "txn_count"
metric_label = "Transactions per person" if per_capita else "Total transactions"

col_m, col_t = st.columns([3, 2])

with col_m:
    gj = load_geojson(A.GEOJSON_URL)
    lookup = A.build_geo_lookup(gj, map_agg["state"]) if gj else {}
    if gj is not None and lookup:
        map_agg["geo_key"] = map_agg["state"].map(lookup)
        matched = map_agg["geo_key"].notna().sum()

        figm = px.choropleth(
            map_agg.dropna(subset=["geo_key"]),
            geojson=gj, featureidkey="properties.ST_NM",
            locations="geo_key", color=metric_col,
            color_continuous_scale="Purples",
            hover_name="State",
            hover_data={"geo_key": False, metric_col: ":,.1f"},
            labels={metric_col: metric_label},
        )
        figm.update_geos(fitbounds="locations", visible=False)
        figm.update_layout(
            title=f"{metric_label} by state ({year_range[0]}–{year_range[1]})",
            margin=dict(t=50, b=10, l=10, r=10), height=460,
            paper_bgcolor="white",
        )
        st.plotly_chart(figm, width="stretch")
        if matched < len(map_agg):
            missing = map_agg.loc[map_agg["geo_key"].isna(), "State"].tolist()
            st.caption(
                f"⚠️ {matched}/{len(map_agg)} states matched the map boundaries. "
                f"Unmatched: {', '.join(missing)}"
            )
    else:
        st.info("Map boundaries unavailable offline — showing ranked bars instead.")
        fb = map_agg.nlargest(top_n_states, metric_col).sort_values(metric_col)
        figfb = go.Figure(go.Bar(
            x=fb[metric_col], y=fb["State"], orientation="h",
            marker_color=PP_PURPLE, opacity=0.85
        ))
        figfb.update_layout(
            title=f"{metric_label} — top {top_n_states}",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=50, l=10), height=460
        )
        st.plotly_chart(figfb, width="stretch")

with col_t:
    rank = map_agg.dropna(subset=["per_capita"]).copy()
    rank["Volume rank"]     = rank["txn_count"].rank(ascending=False).astype(int)
    rank["Per-capita rank"] = rank["per_capita"].rank(ascending=False).astype(int)
    rank["Δ"] = rank["Volume rank"] - rank["Per-capita rank"]
    show = (rank.sort_values("Per-capita rank")
                .head(12)[["State","Volume rank","Per-capita rank","Δ","per_capita"]]
                .rename(columns={"per_capita": "Txns/person"}))
    st.markdown("##### Volume rank vs per-capita rank")
    st.dataframe(
        show, hide_index=True, width="stretch", height=430,
        column_config={
            "Txns/person": st.column_config.NumberColumn(format="%.0f"),
            "Δ": st.column_config.NumberColumn(
                format="%+d", help="Positive = punches above its population weight"),
        }
    )

takeaway(
    "<b>Volume rankings mostly measure population.</b> Normalising per capita moves "
    "Telangana to #1 (286 transactions per person) and drops Maharashtra from #1 to #7. "
    "Uttar Pradesh is the clearest case: 4th by raw volume, far lower per person. "
    "Toggle <i>Normalise per capita</i> in the sidebar to switch the map between the two views."
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 3: State Market Share  |  Transaction Type Mix
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🏙️ State & Category Analysis</div>",
            unsafe_allow_html=True)
col_l2, col_r2 = st.columns([2, 3])

with col_l2:
    state_agg = (filt_states.groupby("state")["txn_count"].sum().reset_index())
    if per_capita:
        state_agg = state_agg.merge(df_pop, on="state", how="left")
        state_agg["metric"] = state_agg["txn_count"] / state_agg["population_2011"]
        axis_title, fmt = "Transactions per person", "{:.0f}"
    else:
        state_agg["metric"] = state_agg["txn_count"] / 1e9
        axis_title, fmt = "Transactions (Billions)", "{:.1f}"

    state_agg = state_agg.nlargest(top_n_states, "metric")
    state_agg["State"] = state_agg["state"].map(A.normalise_state)

    fig3 = go.Figure(go.Bar(
        x=state_agg["metric"], y=state_agg["State"],
        orientation="h", marker_color=PP_PURPLE, opacity=0.85,
        text=state_agg["metric"].apply(fmt.format), textposition="outside"
    ))
    fig3.update_layout(
        title=f"Top {top_n_states} States",
        xaxis_title=axis_title, yaxis_autorange="reversed",
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

takeaway(
    "<b>UPI's original use case is nearly dead.</b> Recharge &amp; bill payments fell from "
    "36.9% of volume (2018) to 5.7% (2024) while merchant payments went 10.4% → 60.8%. "
    "PhonePe's revenue base is now merchant MDR, not transfers — a different business "
    "with different unit economics than the one it launched with."
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 4: Device brands  (previously unused table)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📱 Who Is the PhonePe User?</div>",
            unsafe_allow_html=True)

dev = df_dev[(df_dev["year"] >= year_range[0]) & (df_dev["year"] <= year_range[1])]
col_d1, col_d2 = st.columns([3, 2])

with col_d1:
    brand_tot = (dev.groupby("brand")["user_count"].sum()
                 .sort_values(ascending=False).reset_index())
    brand_tot["share"] = brand_tot["user_count"] / brand_tot["user_count"].sum() * 100
    topb = brand_tot.head(10).sort_values("share")

    figd = go.Figure(go.Bar(
        x=topb["share"], y=topb["brand"], orientation="h",
        marker_color=[PP_ORANGE if b == "Apple" else PP_PURPLE for b in topb["brand"]],
        opacity=0.88,
        text=topb["share"].apply(lambda v: f"{v:.1f}%"), textposition="outside"
    ))
    figd.update_layout(
        title="Device brand share of PhonePe users (top 10)",
        xaxis_title="Share of users (%)",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, l=10, r=60), height=380
    )
    st.plotly_chart(figd, width="stretch")

with col_d2:
    brand_yr = (dev.groupby(["year","brand"])["user_count"].sum().reset_index())
    keep = brand_tot.head(5)["brand"].tolist()
    brand_yr = brand_yr[brand_yr["brand"].isin(keep)]
    brand_yr["share"] = brand_yr.groupby("year")["user_count"].transform(
        lambda s: s / s.sum() * 100)

    figd2 = px.line(
        brand_yr, x="year", y="share", color="brand", markers=True,
        color_discrete_sequence=[PP_PURPLE, PP_GREEN, PP_ORANGE, PP_RED, PP_LIGHT],
        labels={"share": "Share of top-5 users (%)", "year": "Year", "brand": "Brand"},
        title="Brand share over time"
    )
    figd2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3, font_size=10),
        margin=dict(t=50, b=70), height=380
    )
    st.plotly_chart(figd2, width="stretch")

apple_share = brand_tot.loc[brand_tot["brand"] == "Apple", "share"]
apple_txt = f"{apple_share.iloc[0]:.1f}%" if len(apple_share) else "under 1%"
top4 = ", ".join(
    f"{r.brand} ({r.share:.1f}%)"
    for r in brand_tot[brand_tot["brand"] != "Others"].head(4).itertuples()
)
takeaway(
    f"<b>PhonePe is a mass-market Android product, not a premium one.</b> Apple accounts "
    f"for just {apple_txt} of users — the base is {top4}. That constrains monetisation "
    "strategy: this is a user base reached through low-end Android distribution and "
    "volume economics, not high-ARPU premium features."
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 5: User adoption S-curve
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>👥 Adoption S-Curve — One Curve, Different Positions</div>",
            unsafe_allow_html=True)

curve = A.fit_adoption_curve(df_users, pop=df_pop)
col_s1, col_s2 = st.columns([3, 2])

with col_s1:
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
        fillcolor="rgba(95,37,159,0.15)"
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
        margin=dict(t=55, b=30), height=380
    )
    fig5.update_yaxes(title_text="Users (Millions)", secondary_y=False)
    fig5.update_yaxes(title_text="YoY Growth %",     secondary_y=True)
    st.plotly_chart(fig5, width="stretch")

with col_s2:
    if not curve.empty:
        c = curve.copy()
        c["State"] = c["state"].map(A.normalise_state)
        lead, lag = c.head(8), c.tail(8)
        band = pd.concat([lead, lag]).sort_values("quarters_behind_leader")
        figs = go.Figure(go.Bar(
            x=band["quarters_behind_leader"], y=band["State"], orientation="h",
            marker_color=[PP_GREEN if v <= 12 else PP_RED
                          for v in band["quarters_behind_leader"]],
            opacity=0.85,
            text=band["quarters_behind_leader"].apply(lambda v: f"{v:.0f}Q"),
            textposition="outside"
        ))
        figs.update_layout(
            title=f"Quarters behind the leader ({A.normalise_state(curve['leader'].iloc[0])})",
            xaxis_title="Quarters behind", yaxis_autorange="reversed",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=50, l=10, r=50), height=380
        )
        st.plotly_chart(figs, width="stretch")
    else:
        st.info("Not enough data points to fit adoption curves.")

if not curve.empty:
    laggard = A.normalise_state(curve.iloc[-1]["state"])
    gap_q   = curve.iloc[-1]["quarters_behind_leader"]
    takeaway(
        f"<b>Every state is on the same curve, just at different points.</b> Fitting a "
        f"logistic to each state's user penetration and comparing midpoints, "
        f"{laggard} sits {gap_q:.0f} quarters (~{gap_q/4:.1f} years) behind "
        f"{A.normalise_state(curve['leader'].iloc[0])}. This single number explains the "
        "volume gaps, the merchant-mix gaps and the apparent small-state 'anomalies' below "
        "as one phenomenon rather than three."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 6: Merchant-adjusted anomaly detection
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🔍 Anomaly Detection (Merchant-Mix Adjusted)</div>",
            unsafe_allow_html=True)

latest = df_states[df_states["year"] == MAX_YEAR]
anom_base = (latest.groupby("state")
             .agg(txn_count=("txn_count","sum"), txn_amount=("txn_amount","sum"))
             .reset_index())
merch_ct = (latest[latest["transaction_type"] == "Merchant payments"]
            .groupby("state")["txn_count"].sum().rename("merchant_count").reset_index())
anom_base = anom_base.merge(merch_ct, on="state", how="left")
anom_base["merchant_count"] = anom_base["merchant_count"].fillna(0)

res = A.merchant_adjusted_outliers(anom_base)
corr = A.merchant_ticket_correlation(anom_base)

col_a1, col_a2 = st.columns([3, 2])

with col_a1:
    res["State"] = res["state"].map(A.normalise_state)
    cmap = {"ABOVE MODEL": PP_RED, "BELOW MODEL": PP_GREEN, "AS EXPECTED": "#c9c9c9"}
    figa = go.Figure()
    figa.add_trace(go.Scatter(
        x=res["merchant_pct"], y=res["avg_ticket"],
        mode="markers", marker=dict(
            size=11, color=[cmap[f] for f in res["flag"]],
            line=dict(width=1, color="white")),
        text=res["State"], hovertemplate="<b>%{text}</b><br>"
        "Merchant share: %{x:.1f}%<br>Avg ticket: ₹%{y:.0f}<extra></extra>",
        name="States"
    ))
    xs = np.linspace(res["merchant_pct"].min(), res["merchant_pct"].max(), 50)
    slope, intercept = np.polyfit(res["merchant_pct"], res["avg_ticket"], 1)
    figa.add_trace(go.Scatter(
        x=xs, y=intercept + slope * xs, mode="lines",
        line=dict(color=PP_PURPLE, dash="dash", width=2),
        name=f"Fit (r = {corr:.2f})"
    ))
    for _, r in res[res["flag"] != "AS EXPECTED"].iterrows():
        figa.add_annotation(x=r["merchant_pct"], y=r["avg_ticket"],
                            text=r["State"], showarrow=True, arrowhead=2,
                            arrowsize=0.8, font_size=10, ax=25, ay=-18)
    figa.update_layout(
        title=f"Average ticket vs merchant share ({MAX_YEAR})",
        xaxis_title="Merchant payments as % of volume",
        yaxis_title="Average ticket (₹)",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.1), margin=dict(t=55), height=400
    )
    st.plotly_chart(figa, width="stretch")

with col_a2:
    tbl = (res.sort_values("residual_z", ascending=False)
           [["State","merchant_pct","avg_ticket","expected_ticket","residual_z","flag"]]
           .rename(columns={
               "merchant_pct":    "Merchant %",
               "avg_ticket":      "Actual ₹",
               "expected_ticket": "Expected ₹",
               "residual_z":      "Residual z",
               "flag":            "Flag"}))
    st.markdown(f"##### Residuals after controlling for merchant mix ({MAX_YEAR})")
    st.dataframe(
        tbl, hide_index=True, width="stretch", height=370,
        column_config={
            "Merchant %": st.column_config.NumberColumn(format="%.1f%%"),
            "Actual ₹":   st.column_config.NumberColumn(format="₹%.0f"),
            "Expected ₹": st.column_config.NumberColumn(format="₹%.0f"),
            "Residual z": st.column_config.NumberColumn(format="%.2f"),
        }
    )

flagged = res[res["flag"] != "AS EXPECTED"]["State"].tolist()
takeaway(
    f"<b>The naive z-score was measuring merchant penetration, not anomalies.</b> "
    f"Average ticket and merchant share correlate at r = {corr:.2f}, so simply z-scoring "
    "ticket size re-flags every low-merchant state — Manipur, Nagaland and Mizoram were "
    "false positives, not outliers. Regressing ticket on merchant share and scoring the "
    f"<i>residual</i> leaves {len(flagged)} genuine cases: "
    f"{', '.join(flagged) if flagged else 'none'}. These are states that are unusual "
    "<i>relative to peers with the same payment mix</i>."
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROW 7: Merchant vs P2P deep-dive
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
    mp_pivot["State"] = mp_pivot["state"].map(A.normalise_state)
    mp_pivot = mp_pivot.sort_values("merchant_pct", ascending=True)

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        name="Merchant Payments", y=mp_pivot["State"],
        x=mp_pivot["Merchant payments"] / 1e6,
        orientation="h", marker_color=PP_PURPLE, opacity=0.85
    ))
    fig6.add_trace(go.Bar(
        name="P2P Payments", y=mp_pivot["State"],
        x=mp_pivot["Peer-to-peer payments"] / 1e6,
        orientation="h", marker_color=PP_GREEN, opacity=0.85
    ))
    fig6.update_layout(
        barmode="stack",
        title=f"Merchant vs P2P Volume — Top {top_n_states} States (Millions)",
        xaxis_title="Transactions (Millions)",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=55, b=30), height=380
    )
    st.plotly_chart(fig6, width="stretch")

    lo = mp_pivot.iloc[0]
    hi = mp_pivot.iloc[-1]
    takeaway(
        f"<b>Merchant penetration is the leading indicator of maturity.</b> Among the top "
        f"{top_n_states} states, {hi['State']} runs {hi['merchant_pct']:.0f}% merchant "
        f"versus {lo['State']} at {lo['merchant_pct']:.0f}%. States low on this measure "
        "are not underperforming — they are earlier on the same curve, and their merchant "
        "share is the metric to watch for the next leg of volume growth."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:12px'>"
    "Data: PhonePe Pulse (github.com/PhonePe/pulse) &nbsp;|&nbsp; "
    "Population: Census of India 2011, reorganisation-adjusted &nbsp;|&nbsp; "
    "Built by Meet Kumar Sarkar, NIT Patna &nbsp;|&nbsp; "
    "Stack: Python · SQL · Streamlit · Plotly"
    "</div>",
    unsafe_allow_html=True
)
