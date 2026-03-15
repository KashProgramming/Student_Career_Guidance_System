# 🎓 Student Career Guide & Placement Predictor

A comprehensive machine learning system designed to predict campus placement success, estimate expected salaries, and provide actionable career guidance for students. This project integrates a FastAPI backend, a Streamlit frontend, and a complete MLOps pipeline for continuous training and drift monitoring.

## Key Features

### Machine Learning Engine
- **Dual Prediction Architecture**: Simultaneously predicts **Placement Probability** (Classification) and **Expected Salary** (Regression) using optimized models (XGBoost/RandomForest).
- **Risk Assessment**: Automatically categorizes students into **High, Moderate, or Low Risk tiers** based on their placement likelihood.
- **Explainable AI (XAI)**: Leverages **SHAP (SHapley Additive exPlanations)** to break down exactly which factors (e.g., GPA, certifications, internships) helped or hindered a prediction.
- **Dynamic Recommender**: Runs "what-if" simulations to suggest the most impactful improvements a student can make to increase their placement chances.

### Frontend & User Experience
- **Interactive Dashboard**: A sleek Streamlit interface for data input and real-time result visualization.
- **Visual Analytics**: Interactive Waterfall and Bar plots to visualize feature importance for individual students.
- **PDF Report Generation**: Professional career guidance reports generated on-the-fly, including model insights and personalized tips.
- **Trend Analysis**: Tracks historical prediction data to identify patterns over time.

### ⚙️ MLOps & Pipeline
- **Automated Training Pipeline**: End-to-end workflow from data ingestion and validation to feature engineering and model registration.
- **Model Versioning**: Maintains a history of trained models with metadata and performance metrics.
- **Drift Detection**: Advanced monitoring using **Population Stability Index (PSI)** to detect changes in input data distributions.
- **Auto-Retraining**: Automated logic to trigger a fresh model training run if significant data drift is detected, preventing performance decay.

---

## Project Structure

```text
├── backend/                # FastAPI application (Inference & Explainability)
├── frontend/               # Streamlit application (User Interface)
├── ml_pipeline/            # Scripts for training, evaluation, and registration
├── monitoring/             # Drift detection and prediction logging utilities
├── models/                 # Model registry (Current & Versioned artifacts)
├── shared/                 # Common preprocessing logic (Common to Train/Serve)
├── utils/                  # UI helpers, PDF generation, and SHAP plotting
├── docker/                 # Deployment configurations (Dockerfiles)
└── data/                   # Dataset and schema metadata
```

---

## 🛠️ Tech Stack

- **Languages**: Python 3.11
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Streamlit, Plotly, ReportLab
- **ML/Science**: Scikit-Learn, SHAP, Pandas, NumPy, XGBoost
- **DevOps**: Docker, Docker Compose, Git

---

## Getting Started

### Running with Docker (Recommended)
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd student_career_guide_streamlit
   ```

2. **Start the services**:
   ```bash
   docker-compose up --build
   ```

3. **Access the applications**:
   - **Frontend**: [http://localhost:8501](http://localhost:8501)
   - **Backend API (Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Local Development (Manual)
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models and register artifacts**:
   ```bash
   python -m ml_pipeline.train_pipeline --data-version v1
   ```

3. **Start the services**:
   ```bash
   # Terminal 1: Backend
   uvicorn backend.app.main:app --reload --port 8000
   
   # Terminal 2: Frontend
   export API_BASE_URL=http://localhost:8000
   streamlit run frontend/app.py
   ```

---

## 📈 MLOps Workflows

### Model Retraining
To manually trigger a full model training and registration cycle:
```bash
export PYTHONPATH=$PWD
python ml_pipeline/train_pipeline.py
```

### Drift Monitoring
The system logs every live prediction. To check for data drift against the training baseline:
```bash
python -m monitoring.drift_detection --data-version v1
```
If drift is detected (PSI > 0.2), you can run `retrain_on_drift.py` to update the model automatically.

---

## CI/CD
See .github/workflows/mlops.yml for the GitHub Actions pipeline and ci_cd/README.md for details.
