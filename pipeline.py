"""
Step 1 - Data pipeline: raw CSV -> validate -> clean -> load into a database.

Usage:
    python pipeline.py                      # loads into local SQLite (no setup needed)
    set DB_URL=postgresql://user:pass@localhost:5432/frauddb   (Windows)
    python pipeline.py                      # loads into PostgreSQL instead

Input : data/creditcard.csv  (Kaggle: "Credit Card Fraud Detection", ULB)
Output: table `transactions_clean` + table `data_quality_report` + logs/pipeline.log
"""
import logging
import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd

RAW_PATH = os.environ.get("RAW_PATH", "data/creditcard.csv")
DB_URL = os.environ.get("DB_URL")            # empty -> SQLite
SQLITE_PATH = "outputs/fraud.db"

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/pipeline.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


# ---------- EXTRACT ----------
def extract(path: str) -> pd.DataFrame:
    log.info("Extracting %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d rows, %d columns", *df.shape)
    return df


# ---------- VALIDATE ----------
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Run data-quality checks and return a report table (one row per check)."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema check failed, missing columns: {missing_cols}")

    n = len(df)
    checks = [
        ("schema_all_expected_columns_present", 0, "All 31 expected columns found"),
        ("completeness_null_cells", int(df.isna().sum().sum()), "Total null cells"),
        ("uniqueness_duplicate_rows", int(df.duplicated().sum()), "Fully duplicated rows"),
        ("validity_negative_amount", int((df["Amount"] < 0).sum()), "Amount < 0"),
        ("validity_invalid_class_label", int((~df["Class"].isin([0, 1])).sum()), "Class not in {0,1}"),
        ("validity_negative_time", int((df["Time"] < 0).sum()), "Time < 0"),
    ]
    report = pd.DataFrame(checks, columns=["check_name", "failed_rows", "description"])
    report["total_rows"] = n
    report["pct_failed"] = (report["failed_rows"] / n * 100).round(4)
    report["status"] = report["failed_rows"].apply(lambda x: "PASS" if x == 0 else "FAIL")
    report["run_timestamp"] = datetime.now().isoformat(timespec="seconds")
    for _, r in report.iterrows():
        log.info("[%s] %s -> %d failed rows", r["status"], r["check_name"], r["failed_rows"])
    return report


# ---------- TRANSFORM ----------
def transform(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    df = df[(df["Amount"] >= 0) & (df["Time"] >= 0) & (df["Class"].isin([0, 1]))]
    df = df.dropna()
    log.info("Cleaning removed %d rows (%d remain)", before - len(df), len(df))

    # engineered features that are useful for analysis / Power BI
    df["Class"] = df["Class"].astype(int)
    df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
    df["amount_band"] = pd.cut(
        df["Amount"],
        bins=[-0.01, 10, 50, 200, 1000, float("inf")],
        labels=["0-10", "10-50", "50-200", "200-1000", "1000+"],
    ).astype(str)
    df.insert(0, "txn_id", range(1, len(df) + 1))
    return df.reset_index(drop=True)


# ---------- LOAD ----------
def load(df: pd.DataFrame, report: pd.DataFrame) -> None:
    if DB_URL:
        from sqlalchemy import create_engine  # imported only when PostgreSQL is used
        engine = create_engine(DB_URL)
        df.to_sql("transactions_clean", engine, if_exists="replace", index=False, chunksize=10_000)
        report.to_sql("data_quality_report", engine, if_exists="append", index=False)
        log.info("Loaded into database from DB_URL")
    else:
        with sqlite3.connect(SQLITE_PATH) as conn:
            df.to_sql("transactions_clean", conn, if_exists="replace", index=False, chunksize=10_000)
            report.to_sql("data_quality_report", conn, if_exists="append", index=False)
        log.info("Loaded into SQLite at %s", SQLITE_PATH)


def main() -> None:
    start = datetime.now()
    try:
        raw = extract(RAW_PATH)
        report = validate(raw)
        clean = transform(raw)
        load(clean, report)
        log.info("Pipeline finished OK in %.1fs", (datetime.now() - start).total_seconds())
    except Exception:
        log.exception("Pipeline FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
