# Daily Demand Forecasting Orders — FastAPI Backend

A clean, minimal, beginner-friendly REST API built with **FastAPI** that serves real-time predictions for daily order demand using a pre-trained scikit-learn regression pipeline.

---

## 1. Project Overview

This service exposes a machine learning model trained to forecast total daily demand (orders) from logistics, traffic controller, banking, and sector-specific operational indicators.

- **Framework**: FastAPI (Python)
- **Model Pipeline**: Scikit-Learn Pipeline (`ColumnTransformer` with `OneHotEncoder` & `StandardScaler` + `LinearRegression`)
- **Serialized Artifact**: `demand_regression_modelAA.joblib`

---

## 2. Project Structure

```text
Daily Demand Forecasting Orders/
├── app/
│   ├── __init__.py           # Package marker for the app module
│   ├── main.py               # FastAPI application, lifespan handler, and route endpoints
│   └── schemas.py            # Pydantic schemas for request validation and response formatting
├── demand_regression_modelAA.joblib  # Pre-trained ML pipeline artifact
├── Daily Demand Forecasting Orders.csv # Original dataset
├── daily_demand_forecasting (1).ipynb # Model training & evaluation notebook
├── requirements.txt          # Minimal Python dependencies
├── .gitignore                # Git ignore rules for virtual environments, caches, etc.
└── README.md                 # Setup guide and API documentation
```

### Purpose of Backend Files

- **`app/__init__.py`**: Initializes `app` as an importable Python package.
- **`app/schemas.py`**: Defines `OrderPredictionRequest`, `PredictionResponse`, and `HealthResponse` using Pydantic to validate input data types and field boundaries.
- **`app/main.py`**: Configures the FastAPI app, loads the trained model artifact into memory once during application startup, and routes incoming HTTP requests to endpoints.

---

## 3. Setup and Installation (Windows)

### Step 1: Open PowerShell or Command Prompt
Open your terminal and navigate to the project directory:
```powershell
cd "c:\Users\yashu\OneDrive\Desktop\Daily Demand Forecasting Orders"
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**In PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```
*(If you see an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

**In Command Prompt (cmd):**
```cmd
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 4. Running the API

Start the local development server using Uvicorn from the project root:

```powershell
python -m uvicorn app.main:app --reload
```

By default, the server runs at: `http://127.0.0.1:8000`

*(Note: If port 8000 is in use by another program, run `python -m uvicorn app.main:app --reload --port 8001`)*

---

## 5. Interactive API Documentation (Swagger UI)

FastAPI automatically generates an interactive Swagger UI. Once the server is running, open your web browser and visit:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc alternative**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

You can test all endpoints directly inside the browser using the **Try it out** button.

---

## 6. Endpoints Reference

### 1. `GET /`
Verifies that the API service is up and running.

**Sample Response (`200 OK`):**
```json
{
  "message": "Daily Demand Forecasting API is running",
  "docs_url": "/docs",
  "health_url": "/health"
}
```

---

### 2. `GET /health`
Verifies that the application and the machine learning model are ready to serve predictions.

**Sample Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 3. `POST /predict`
Generates a forecast of total daily orders using operational indicators.

#### Field Descriptions:
- `Week of the month (first week, second, third, fourth or fifth week)`: Integer from `1` to `5`.
- `Day of the week (Monday to Friday)`: Integer from `2` to `6` (encoded as: Monday = 2, Tuesday = 3, Wednesday = 4, Thursday = 5, Friday = 6).
- `Non-urgent order`: Numerical value (≥ 0).
- `Urgent order`: Numerical value (≥ 0).
- `Fiscal sector orders`: Numerical value (≥ 0).
- `Orders from the traffic controller sector`: Integer or float (≥ 0).
- `Banking orders (1)`: Integer or float (≥ 0).
- `Banking orders (2)`: Integer or float (≥ 0).
- `Banking orders (3)`: Integer or float (≥ 0).

#### Example Request:
```json
{
  "Week of the month (first week, second, third, fourth or fifth week": 1,
  "Day of the week (Monday to Friday)": 4,
  "Non-urgent order": 316.307,
  "Urgent order": 223.27,
  "Fiscal sector orders": 0.0,
  "Orders from the traffic controller sector": 65556,
  "Banking orders (1)": 44914,
  "Banking orders (2)": 188411,
  "Banking orders (3)": 14793,
  "include_step_3": true
}
```

#### Example Response (`200 OK`):
```json
{
  "forecast_step_1": 271.1402,
  "forecast_step_2": 263.9835,
  "forecast_step_3": 315.9324,
  "predicted_total_orders": 271.1402,
  "status": "success",
  "observation_horizon": "Next observation steps (t+1, t+2, t+3)"
}
```

---

### 4. `POST /forecast`
Dedicated future multi-step forecasting endpoint returning Next Step 1 ($t+1$), Next Step 2 ($t+2$), and optional Next Step 3 ($t+3$). Takes the same payload as `/predict`.

---

## 7. Model Retraining and Multi-Step Formulation

- **Why Retraining Was Necessary:** The legacy pipeline (`demand_regression_modelAA.joblib`) was trained for same-day prediction ($X_t \to y_t$) where same-day urgent and non-urgent orders directly approximate the target. Future operational metrics $X_{t+1}, X_{t+2}$ are unavailable at time $t$. Simply relabeling same-day predictions would be methodologically invalid.
- **Multi-Step Pipeline:** The script `train_future_forecast_model.py` fits a multi-output pipeline on sequential observation steps ($y_{t+1}, y_{t+2}, y_{t+3}$) without leaking `Order type A/B/C`, saving the artifact to `demand_forecast_multistep_model.joblib`.
- **Legacy Preservation:** The original artifact `demand_regression_modelAA.joblib` remains untouched.

---

## 8. Safe Deployment Update Guide (Render)

1. **Review and Test Locally:**
   ```powershell
   pytest tests/
   python -m uvicorn app.main:app --reload
   streamlit run streamlit_app.py
   ```
2. **Commit Feature Branch:**
   Commit the changes on the `feature/future-forecasting` branch.
3. **Merge Pull Request / Branch:**
   When reviewed and approved, merge `feature/future-forecasting` into `main`.
4. **Render Automatic Build:**
   Render will detect the commit to `main`, install dependencies from `requirements.txt`, and start `uvicorn app.main:app`. The new model artifact `demand_forecast_multistep_model.joblib` will be loaded on startup.

