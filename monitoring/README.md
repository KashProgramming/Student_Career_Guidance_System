# Monitoring

- Prediction logs are stored at monitoring/logs/predictions.jsonl.
- Run drift detection with:

```bash
python -m monitoring.drift_detection --data-version v1
```

- Generate a quick latency summary with:

```bash
python -m monitoring.dashboard
```

- Trigger retraining on drift:

```bash
python -m monitoring.retrain_on_drift --data-version v1 --psi-threshold 0.2
```
