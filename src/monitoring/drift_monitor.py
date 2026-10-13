"""
Drift Monitor — Phase 11
=========================
Logs every search request and generates Evidently AI drift reports.

Tracks three signals over time:
  1. Score drift       — are result quality scores changing?
  2. Latency drift     — is the system getting slower?
  3. Category drift    — are users searching different categories?

Usage:
    from src.monitoring.drift_monitor import DriftMonitor
    monitor = DriftMonitor()
    monitor.log(query="AI video generator", results=[...], latency_ms=120.5)
    monitor.generate_report()   # saves data/drift_report.html
"""

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
LOG_PATH   = ROOT / "data" / "query_log.jsonl"
REPORT_DIR = ROOT / "data"

# Minimum entries needed to split into reference/current and run a report
MIN_ENTRIES_FOR_REPORT = 20


class DriftMonitor:
    """
    Append-only query logger + Evidently drift reporter.

    Each log entry (one JSON line) captures:
      ts            float   Unix timestamp
      query         str     original user query
      corrected     bool    was the query spell-corrected?
      n_results     int     how many results returned
      top_score     float   highest final_score in results
      mean_score    float   mean final_score across results
      latency_ms    float   end-to-end API latency
      top_category  str     category of the #1 result
    """

    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────────────

    def log(
        self,
        query: str,
        results: list[dict],
        latency_ms: float,
        corrected: bool = False,
    ) -> None:
        """Append one request to the query log."""
        scores = [r.get("final_score", 0.0) for r in results]
        entry = {
            "ts":           time.time(),
            "query":        query,
            "corrected":    corrected,
            "n_results":    len(results),
            "top_score":    round(max(scores, default=0.0), 4),
            "mean_score":   round(sum(scores) / len(scores), 4) if scores else 0.0,
            "latency_ms":   round(latency_ms, 1),
            "top_category": results[0].get("category", "Unknown") if results else "Unknown",
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.warning(f"Failed to write to query log: {e}")

    # ── Load log ───────────────────────────────────────────────────────────────

    def load_log(self) -> pd.DataFrame:
        """Load the full query log into a DataFrame."""
        if not self.log_path.exists():
            return pd.DataFrame()

        rows = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
        return df

    def n_entries(self) -> int:
        if not self.log_path.exists():
            return 0
        with open(self.log_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # ── Drift report ───────────────────────────────────────────────────────────

    def generate_report(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Generate an Evidently drift report comparing:
          reference = first half of logged queries
          current   = second half (most recent)

        Saves HTML to data/drift_report.html and returns the path.
        Returns None if there aren't enough entries yet.
        """
        df = self.load_log()
        n  = len(df)

        if n < MIN_ENTRIES_FOR_REPORT:
            log.warning(
                f"Only {n} log entries — need at least {MIN_ENTRIES_FOR_REPORT} "
                "to generate a drift report."
            )
            return None

        # Split into reference (older) and current (newer)
        mid       = n // 2
        reference = df.iloc[:mid].copy()
        current   = df.iloc[mid:].copy()

        # Columns to monitor for drift
        numerical_cols    = ["top_score", "mean_score", "latency_ms", "n_results"]
        categorical_cols  = ["top_category"]

        # Keep only numeric + categorical columns (drop ts, query strings)
        ref_data = reference[numerical_cols + categorical_cols].reset_index(drop=True)
        cur_data = current[numerical_cols + categorical_cols].reset_index(drop=True)

        # Ensure correct dtypes
        for col in numerical_cols:
            ref_data[col] = ref_data[col].astype(float)
            cur_data[col] = cur_data[col].astype(float)

        try:
            from evidently import Report, Dataset, DataDefinition
            from evidently.presets import DataDriftPreset

            dd = DataDefinition(
                numerical_columns=numerical_cols,
                categorical_columns=categorical_cols,
            )

            report  = Report([DataDriftPreset()])
            snapshot = report.run(
                reference_data=Dataset.from_pandas(ref_data, data_definition=dd),
                current_data=Dataset.from_pandas(cur_data, data_definition=dd),
            )

            out = output_path or (REPORT_DIR / "drift_report.html")
            snapshot.save_html(str(out))
            log.info(f"Drift report saved to {out}")
            return out

        except Exception as e:
            log.error(f"Evidently report failed: {e}")
            return None

    # ── Quick summary ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a quick stats summary without running Evidently."""
        df = self.load_log()
        if df.empty:
            return {"n_entries": 0}

        top_cats = Counter(df["top_category"].tolist()).most_common(5)

        return {
            "n_entries":        len(df),
            "mean_top_score":   round(df["top_score"].mean(), 4),
            "mean_latency_ms":  round(df["latency_ms"].mean(), 1),
            "p95_latency_ms":   round(df["latency_ms"].quantile(0.95), 1),
            "cache_hit_rate":   round(df["corrected"].mean(), 3),
            "top_categories":   [c for c, _ in top_cats],
            "date_range":       {
                "from": str(df["ts"].min()),
                "to":   str(df["ts"].max()),
            },
        }
