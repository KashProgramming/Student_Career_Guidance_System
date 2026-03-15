from __future__ import annotations

import os
from typing import Dict

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def post(endpoint: str, payload: Dict) -> Dict:
    url = f"{API_BASE_URL}{endpoint}"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def get(endpoint: str) -> Dict:
    url = f"{API_BASE_URL}{endpoint}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
