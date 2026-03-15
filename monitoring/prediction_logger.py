from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

LOG_PATH = Path(__file__).resolve().parents[1] / "monitoring" / "logs" / "predictions.jsonl"


def load_prediction_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()

    records: List[dict] = []
    with LOG_PATH.open("r") as handle:
        for line in handle:
            records.append(json.loads(line))

    return pd.json_normalize(records)
