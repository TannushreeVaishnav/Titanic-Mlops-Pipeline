# Titanic Survival Prediction — MLOps Project

> **End-to-end MLOps pipeline** featuring ETL with Apache Airflow, a Redis Feature Store, data drift detection, and real-time ML monitoring via Prometheus + Grafana — built on the classic Titanic dataset.

---

## Note

Don't skip this just because of the dataset title. The **Titanic dataset is just a vehicle** — the real focus is the MLOps infrastructure built around it:

- Automated ETL pipelines with **Apache Airflow**
- Scalable **Feature Store** using Redis
- **Data Drift Detection** (Kolmogorov-Smirnov test)
- Real-time **ML Monitoring** with Prometheus metrics + Grafana dashboards

---

## Architecture Overview

```
Raw CSV
  │
  ▼
┌───────────────────────────────────┐
│  ETL Pipeline (Apache Airflow)    │
│  Extract → Transform → Load       │
└────────────┬──────────────────────┘
             │
     ┌───────▼────────┐
     │  PostgreSQL DB  │  ← Docker container, viewed via DBeaver
     └───────┬─────────┘
             │
     ┌───────▼────────┐
     │  Data Ingestion │  ← psycopg3
     └───────┬─────────┘
             │
     ┌───────▼────────────────┐
     │  Redis Feature Store   │  ← Feature engineering & storage
     └───────┬────────────────┘
             │
     ┌───────▼────────────────┐
     │  Model Training        │  ← Random Forest (sklearn)
     │  (Feature Extraction)  │
     └───────┬────────────────┘
             │
     ┌───────▼────────────────────────────────┐
     │  Flask App (localhost:5000)             │
     │  ├── Prediction Endpoint (/predict)     │
     │  ├── KS Drift Detection                 │
     │  └── Prometheus Metrics (/metrics)      │
     └───────┬────────────────────────────────┘
             │
     ┌───────▼────────────────┐
     │  Grafana Dashboard     │  ← Real-time monitoring
     └────────────────────────┘
```

---

## Highlights

| Feature | Technology |
|---|---|
| ETL Pipeline | Apache Airflow |
| Database | PostgreSQL (Docker) + DBeaver |
| Data Ingestion | psycopg3 |
| Feature Store | Redis |
| Model | Random Forest (scikit-learn) |
| Data Versioning | DVC |
| Drift Detection | Kolmogorov-Smirnov (KSDrift) |
| Serving | Flask |
| Metrics | Prometheus |
| Monitoring | Grafana |

---

## Project Structure

```
titanic-mlops/
├── artifacts/
│   └── model/
│       └── random_forest/
│           └── model.pkl
├── data/
│   └── titanic.csv
├── dags/                          
│   └── titanic_etl_dag.py
├── notebooks/                     
├── src/
│   ├── feature_store.py           
│   ├── logger.py
│   └── ...
├── static/                        
├── templates/
│   └── index.html                 
├── application.py                 
├── docker-compose.yml             
├── requirements.txt
└── README.md
```

---

##  Workflow & Pipeline Steps

### 1. Database Setup
- PostgreSQL runs in a **Docker container**
- Use **DBeaver** to connect and inspect data:
  ```sql
  SELECT * FROM titanic t;
  ```

### 2. ETL Pipeline (Apache Airflow)
- **Extract**: Read `titanic.csv` from the local `data/` folder
- **Transform**: Clean and engineer features
- **Load**: Write processed data into PostgreSQL

### 3. Data Ingestion
- Uses **psycopg3** to pull records from PostgreSQL into Python for downstream processing

### 4. Feature Store (Redis)
- Engineered features are stored as key-value records in Redis
- `RedisFeatureStore` supports:
  - `get_all_entity_ids()` — retrieve all passenger IDs
  - `get_batch_features(entity_ids)` — fetch feature vectors in bulk

### 5. Model Training
- Features are extracted from the Redis store
- A **Random Forest** classifier is trained and serialized to `artifacts/model/random_forest/model.pkl`
- Features used:

  | Feature | Description |
  |---|---|
  | `Pclass` | Passenger class (1, 2, 3) |
  | `Sex` | Male=1, Female=0 |
  | `Age` | Age in years |
  | `Fare` | Ticket fare |
  | `Embarked` | Port (Southampton=0, Cherbourg=1, Queenstown=2) |
  | `FamilySize` | SibSp + Parch |
  | `Isalone` | Travelling alone (Yes=1, No=0) |
  | `HasCabin` | Cabin assigned (Yes=1, No=0) |
  | `Title` | Mr=1, Mrs=2, Miss=3, Master=4 |
  | `Pclass_Fare` | Pclass × Fare (interaction) |
  | `Age_Fare` | Age × Fare (interaction) |

### 6. Data & Code Versioning
- **DVC** (Data Version Control) is used for tracking datasets and model artifacts

### 7. Flask Application
- Served at `http://localhost:5000`
- **Prediction UI** (`/`) — input passenger attributes and run the model
- **Metrics endpoint** (`/metrics`) — Prometheus-format plain text; starts at 0, increments after each prediction

### 8. Data Drift Detection
- On every prediction request, incoming features are scaled with `StandardScaler` and compared against historical Redis data using the **Kolmogorov-Smirnov test**
- If any feature's p-value < 0.05 → drift is flagged and logged

### 9. ML Monitoring
- **Prometheus** scrapes metrics from `:8000` and `/metrics`
- **Grafana** visualises the metrics on a live dashboard

---

## Prometheus Metrics

| Metric | Description |
|---|---|
| `prediction_count_total` | Total number of predictions made |
| `drift_count_total` | Number of times data drift was detected |
| `survived_count_total` | Predictions classified as "Survived" |
| `not_survived_count_total` | Predictions classified as "Did Not Survive" |

---

## Grafana Dashboard

The Grafana dashboard (see screenshot) shows:

- **Prediction Count** histogram — distribution of prediction activity over time
- **Drift Count** time series — drift detection events spike as input distribution shifts
- **Survived / Not Survived** stat panels — real-time outcome counters
- **Scrape Duration** — Prometheus scrape health

---

## Application UI

The Research Workbench UI (at `localhost:5000`) provides:

- **Demographics** — Title, Age, Sex
- **Socio-Economic** — Passenger Class, Fare, Embarked port
- **Travel Details** — Family Size, Is Alone, Has Cabin
- **Advanced Features** — auto-computed `Pclass_Fare` and `Age_Fare` interaction terms
- **Prediction Result** panel — "Survived" (green) / "Not Survived" (dark)
- **Feature Importance** panel — Class & Fare (HIGH), Sex & Age (HIGH), Family Size (MODERATE)
- **Encoding Guide** — inline reference for categorical encodings

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/titanic-mlops.git
cd titanic-mlops
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Infrastructure (PostgreSQL + Redis)

```bash
docker-compose up -d
```

### 4. Trigger the Airflow ETL DAG

Access the Airflow UI and trigger `titanic_etl_dag` to extract, transform, and load data into PostgreSQL.

### 5. Populate the Feature Store

```bash
python src/feature_store.py
```

### 6. Train the Model

```bash
python src/train.py
```

### 7. Run the Flask Application

```bash
python application.py
```

- App UI → `http://localhost:5000`
- Prometheus metrics → `http://localhost:5000/metrics`
- Prometheus server → `http://localhost:8000`

### 8. Set Up Grafana

1. Add Prometheus as a data source (`http://localhost:9090`)
2. Import the dashboard JSON (located in `monitoring/grafana_dashboard.json`)
3. View live metrics

---

## Data Drift Detection — How It Works

```python
# On every /predict request:
features_scaled = scaler.transform(features)   # StandardScaler fitted on Redis historical data
drift = ksd.predict(features_scaled)            # KS test per feature vs. reference distribution

if is_drift == 1:
    drift_count.inc()   # Prometheus counter incremented
    logger.info("Drift Detected.")
```

The `KSDrift` class runs a two-sample KS test column-by-column. If **any** feature's p-value falls below the threshold (default `p_val=0.05`), drift is declared for that request.

---

## Dataset

The Titanic dataset used in this project is publicly available.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| Web Framework | Flask |
| ML | scikit-learn (RandomForestClassifier) (GridSearchCV)|
| Feature Store | Redis |
| Database | PostgreSQL |
| Orchestration | Apache Airflow |
| Containerisation | Docker / Docker Compose |
| DB Client | DBeaver |
| Data Versioning | DVC |
| Drift Detection | SciPy (KS test) |
| Metrics | Prometheus |
| Monitoring | Grafana |

---

## Output
### Prediction UI — Survived
![Prediction UI-Survived](<Screenshot 2026-05-09 095034.png>)
### Prediction UI — Not Survived
![Not Survived](<Screenshot 2026-05-09 091524.png>)
Grafana Dashboard
![Grafana](<Screenshot 2026-05-09 112501.png>)  

---

## Acknowledgements

- Titanic dataset — Kaggle / public domain
- scikit-learn, Flask, Redis, Apache Airflow, Prometheus, Grafana open-source communities
