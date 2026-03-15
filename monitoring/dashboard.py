from __future__ import annotations

import pandas as pd

from monitoring.prediction_logger import load_prediction_logs


def build_summary() -> pd.DataFrame:
    df = load_prediction_logs()
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["latency_ms"] = df["latency_ms"].astype(float)

    summary = {
        "total_requests": len(df),
        "avg_latency_ms": df["latency_ms"].mean(),
        "p95_latency_ms": df["latency_ms"].quantile(0.95),
    }

    return pd.DataFrame([summary])


if __name__ == "__main__":
    print(build_summary())
