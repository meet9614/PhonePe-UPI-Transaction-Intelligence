-- =============================================================================
-- PhonePe UPI Transaction Intelligence — SQL Analysis Queries
-- Dataset  : PhonePe Pulse (github.com/PhonePe/pulse), 2018–2024
-- DB       : SQLite (same syntax runs on MySQL 8+ / PostgreSQL 13+ with minor changes)
-- Author   : Meet Kumar Sarkar, NIT Patna
-- =============================================================================


-- =============================================================================
-- Q1  Quarter-over-Quarter (QoQ) Transaction Growth
--     Technique: CTE + LAG window function
-- =============================================================================
WITH quarterly_totals AS (
    SELECT
        year,
        quarter,
        year || '-Q' || quarter                       AS period,
        SUM(txn_count)                                AS total_txns,
        ROUND(SUM(txn_amount) / 1e9, 2)              AS total_amount_bn
    FROM agg_transactions
    WHERE state = 'india'
    GROUP BY year, quarter
),
with_lag AS (
    SELECT *,
        LAG(total_txns)      OVER (ORDER BY year, quarter) AS prev_txns,
        LAG(total_amount_bn) OVER (ORDER BY year, quarter) AS prev_amount_bn
    FROM quarterly_totals
)
SELECT
    period,
    total_txns,
    total_amount_bn,
    ROUND(100.0 * (total_txns - prev_txns)
          / NULLIF(prev_txns, 0), 1)                       AS txn_count_qoq_pct,
    ROUND(100.0 * (total_amount_bn - prev_amount_bn)
          / NULLIF(prev_amount_bn, 0), 1)                  AS amount_qoq_pct
FROM with_lag
ORDER BY year, quarter;


-- =============================================================================
-- Q2  State-Level Market Share with Ranking
--     Technique: Aggregate window function (SUM OVER) + RANK
-- =============================================================================
WITH latest_year AS (
    SELECT MAX(year) AS yr FROM agg_transactions
),
state_totals AS (
    SELECT
        t.state,
        SUM(t.txn_count)                               AS total_txns,
        ROUND(SUM(t.txn_amount) / 1e9, 2)             AS total_amount_bn,
        ROUND(
            100.0 * SUM(t.txn_count)
            / SUM(SUM(t.txn_count)) OVER (), 2
        )                                               AS market_share_pct
    FROM agg_transactions t
    JOIN latest_year l ON t.year = l.yr
    WHERE t.state != 'india'
    GROUP BY t.state
)
SELECT
    RANK() OVER (ORDER BY total_txns DESC) AS rnk,
    state,
    total_txns,
    total_amount_bn,
    market_share_pct
FROM state_totals
ORDER BY rnk
LIMIT 10;


-- =============================================================================
-- Q3  Transaction-Type Mix Shift  (2018 → 2024)
--     Technique: Conditional aggregation (CASE inside SUM) — pivot in pure SQL
--     Business insight: tracks PhonePe's strategic shift from P2P → Merchant
-- =============================================================================
SELECT
    year,
    ROUND(100.0 * SUM(CASE WHEN transaction_type = 'Peer-to-peer payments'
                           THEN txn_count ELSE 0 END) / SUM(txn_count), 1) AS p2p_pct,
    ROUND(100.0 * SUM(CASE WHEN transaction_type = 'Merchant payments'
                           THEN txn_count ELSE 0 END) / SUM(txn_count), 1) AS merchant_pct,
    ROUND(100.0 * SUM(CASE WHEN transaction_type = 'Recharge & bill payments'
                           THEN txn_count ELSE 0 END) / SUM(txn_count), 1) AS recharge_pct,
    ROUND(100.0 * SUM(CASE WHEN transaction_type = 'Financial Services'
                           THEN txn_count ELSE 0 END) / SUM(txn_count), 1) AS fin_svc_pct
FROM agg_transactions
WHERE state = 'india'
GROUP BY year
ORDER BY year;


-- =============================================================================
-- Q4  Year-over-Year User Growth with Cumulative Running Total
--     Technique: LAG + SUM OVER (ROWS UNBOUNDED PRECEDING) compound window
-- =============================================================================
WITH yearly_users AS (
    SELECT
        year,
        MAX(registered_users) AS peak_registered_users,
        MAX(app_opens)        AS peak_app_opens
    FROM agg_users
    WHERE state = 'india'
    GROUP BY year
),
with_lag AS (
    SELECT *,
        LAG(peak_registered_users) OVER (ORDER BY year)       AS prev_year_users,
        SUM(peak_registered_users) OVER (
            ORDER BY year
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                       AS cumulative_users
    FROM yearly_users
)
SELECT
    year,
    peak_registered_users,
    ROUND(
        100.0 * (peak_registered_users - prev_year_users)
        / NULLIF(prev_year_users, 0), 1
    )                       AS yoy_growth_pct,
    peak_app_opens,
    cumulative_users
FROM with_lag
ORDER BY year;


-- =============================================================================
-- Q5  Anomaly Detection — States with Abnormal Avg Transaction Value
--     Technique: Z-score computed in SQL via manual STDDEV (SQLite-compatible)
--     Business insight: flags potential fraud clusters or data-quality issues
-- =============================================================================
WITH state_metrics AS (
    SELECT
        state,
        SUM(txn_count)  AS total_txn_count,
        SUM(txn_amount) AS total_txn_amount,
        ROUND(SUM(txn_amount) / NULLIF(SUM(txn_count), 0), 0) AS avg_txn_value
    FROM agg_transactions
    WHERE state != 'india' AND year = 2024
    GROUP BY state
),
national_stats AS (
    SELECT
        AVG(avg_txn_value)  AS mean_val,
        -- Population std dev: sqrt(E[X²] - E[X]²)
        SQRT(
            AVG(avg_txn_value * avg_txn_value)
            - AVG(avg_txn_value) * AVG(avg_txn_value)
        )                   AS std_val
    FROM state_metrics
)
SELECT
    m.state,
    m.avg_txn_value,
    ROUND((m.avg_txn_value - s.mean_val) / NULLIF(s.std_val, 0), 2) AS z_score,
    CASE
        WHEN (m.avg_txn_value - s.mean_val) / NULLIF(s.std_val, 0) >  1.5
            THEN 'HIGH_VALUE_OUTLIER'
        WHEN (m.avg_txn_value - s.mean_val) / NULLIF(s.std_val, 0) < -1.5
            THEN 'LOW_VALUE_OUTLIER'
        ELSE 'NORMAL'
    END AS anomaly_flag
FROM state_metrics m CROSS JOIN national_stats s
ORDER BY z_score DESC;


-- =============================================================================
-- Q6  Merchant Payment Penetration Leaders
--     Technique: Multi-condition CASE aggregation + RANK window function
--     Business insight: which states are furthest in the P2P → commerce shift
-- =============================================================================
WITH state_type_breakdown AS (
    SELECT
        state,
        SUM(CASE WHEN transaction_type = 'Merchant payments'
                 THEN txn_count ELSE 0 END) AS merchant_txns,
        SUM(CASE WHEN transaction_type = 'Peer-to-peer payments'
                 THEN txn_count ELSE 0 END) AS p2p_txns,
        SUM(txn_count)                       AS total_txns
    FROM agg_transactions
    WHERE state != 'india' AND year = 2024
    GROUP BY state
),
ranked AS (
    SELECT
        state,
        merchant_txns,
        p2p_txns,
        total_txns,
        ROUND(100.0 * merchant_txns / NULLIF(total_txns, 0), 1) AS merchant_pct,
        RANK() OVER (
            ORDER BY merchant_txns * 1.0 / NULLIF(total_txns, 0) DESC
        )                                                          AS merchant_rank
    FROM state_type_breakdown
)
SELECT
    merchant_rank,
    state,
    merchant_pct,
    merchant_txns,
    p2p_txns
FROM ranked
ORDER BY merchant_rank
LIMIT 10;


-- =============================================================================
-- Q7  4-Quarter Rolling Moving Average  (Smoothed Growth Trend)
--     Technique: AVG OVER (ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
--     Business insight: removes seasonal noise to show true UPI adoption curve
-- =============================================================================
WITH quarterly AS (
    SELECT
        year,
        quarter,
        year || '-Q' || quarter  AS period,
        SUM(txn_count)           AS total_txns,
        ROUND(SUM(txn_amount) / 1e9, 1) AS amount_bn
    FROM agg_transactions
    WHERE state = 'india'
    GROUP BY year, quarter
)
SELECT
    period,
    ROUND(total_txns / 1e6, 1)                    AS txns_millions,
    ROUND(AVG(total_txns) OVER (
        ORDER BY year, quarter
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) / 1e6, 1)                                    AS txn_4q_moving_avg_millions,
    amount_bn,
    ROUND(AVG(amount_bn) OVER (
        ORDER BY year, quarter
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 1)                                           AS amount_4q_moving_avg_bn
FROM quarterly
ORDER BY year, quarter;
