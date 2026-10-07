-- Useful analysis queries (SQLite / PostgreSQL). Practise explaining each one.

-- 1) Fraud rate overall
SELECT COUNT(*) AS txns, SUM("Class") AS fraud_txns,
       ROUND(100.0 * SUM("Class") / COUNT(*), 4) AS fraud_pct
FROM transactions_clean;

-- 2) Fraud by hour of day
SELECT hour_of_day, COUNT(*) AS txns, SUM("Class") AS fraud_txns,
       ROUND(100.0 * SUM("Class") / COUNT(*), 3) AS fraud_pct
FROM transactions_clean
GROUP BY hour_of_day
ORDER BY fraud_pct DESC;

-- 3) Fraud by amount band
SELECT amount_band, COUNT(*) AS txns, SUM("Class") AS fraud_txns,
       ROUND(AVG(Amount), 2) AS avg_amount
FROM transactions_clean
GROUP BY amount_band;

-- 4) Rank hours by fraud count (window function)
WITH h AS (
  SELECT hour_of_day, SUM("Class") AS fraud_txns FROM transactions_clean GROUP BY hour_of_day
)
SELECT hour_of_day, fraud_txns, RANK() OVER (ORDER BY fraud_txns DESC) AS fraud_rank FROM h;

-- 5) Data quality history
SELECT run_timestamp, check_name, failed_rows, status FROM data_quality_report ORDER BY run_timestamp DESC;
