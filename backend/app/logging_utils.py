from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from backend.app.config import LOG_DIR

PREDICTION_LOG = LOG_DIR / "predictions.jsonl"


def log_prediction(payload: Dict) -> None:
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with PREDICTION_LOG.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
