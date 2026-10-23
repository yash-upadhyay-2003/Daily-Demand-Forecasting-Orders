"""
FastAPI application for Daily Demand Forecasting.
Serves multi-step future demand forecasts (Next Step 1, Next Step 2, and optional Next Step 3)
using a pre-trained multi-output regression pipeline.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

# Scikit-learn compatibility shims:
# Ensures models serialized across scikit-learn versions load seamlessly.
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

import sklearn.linear_model as _lm
if not hasattr(_lm.LinearRegression, "tol"):
    _lm.LinearRegression.tol = None

from app.schemas import (
    ForecastResponse,
    HealthResponse,
    OrderForecastRequest,
    OrderPredictionRequest,
    PredictionResponse,
)


# ---------------------------------------------------------------------------
# Configuration & Model Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FORECAST_MODEL_PATH = BASE_DIR / "demand_forecast_multistep_model.joblib"
LEGACY_MODEL_PATH = BASE_DIR / "demand_regression_modelAA.joblib"


# ---------------------------------------------------------------------------
# Model Lifecycle (Loaded once at startup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load ML models once during startup
    and cleanly release resources during shutdown.
    """
    app.state.forecast_model = None
    app.state.legacy_model = None
    app.state.model = None

    # 1. Load multi-step future forecasting model
    if FORECAST_MODEL_PATH.exists():
        try:
            app.state.forecast_model = joblib.load(FORECAST_MODEL_PATH)
            app.state.model = app.state.forecast_model
            print(f"[INFO] Successfully loaded multi-step forecast pipeline from: {FORECAST_MODEL_PATH}")
        except Exception as exc:
            print(f"[ERROR] Failed to load forecast model artifact: {exc}")
    else:
        print(f"[WARNING] Forecast model artifact not found at: {FORECAST_MODEL_PATH}")

    # 2. Check legacy model for safe fallback / inspection
    if LEGACY_MODEL_PATH.exists():
        try:
            app.state.legacy_model = joblib.load(LEGACY_MODEL_PATH)
            # If forecast model was missing, fall back to legacy model
            if app.state.model is None:
                app.state.model = app.state.legacy_model
                print(f"[WARN] Using legacy model fallback from: {LEGACY_MODEL_PATH}")
            else:
                print(f"[INFO] Legacy regression model also verified at: {LEGACY_MODEL_PATH}")
        except Exception as exc:
            print(f"[WARN] Legacy model could not be loaded: {exc}")

    yield

    # Clean up on shutdown
    app.state.forecast_model = None
    app.state.legacy_model = None
    app.state.model = None


# Initialize FastAPI app with descriptive metadata
app = FastAPI(
    title="Daily Demand Multi-Step Forecasting API",
    description=(
        "REST API for forecasting future total orders across observation steps "
        "(Next Step 1, Next Step 2, and optional Next Step 3) based on operational and sector demand metrics."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """Root endpoint confirming that the API service is up and running."""
    return {
        "message": "Daily Demand Future Forecasting API is running",
        "docs_url": "/docs",
        "health_url": "/health",
        "forecast_url": "/forecast",
        "predict_url": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint reporting application and forecasting model readiness."""
    has_model = getattr(app.state, "model", None) is not None
    has_forecast_model = getattr(app.state, "forecast_model", None) is not None

    if not has_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not ready or failed to load.",
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        forecast_model_ready=has_forecast_model,
    )


# ---------------------------------------------------------------------------
# Forecasting Engine Helper
# ---------------------------------------------------------------------------

def _execute_forecast(payload: OrderPredictionRequest) -> ForecastResponse:
    """
    Executes future multi-step forecasting:
    Produces forecasts for Next Step 1 (t+1), Next Step 2 (t+2), and optional Next Step 3 (t+3).
    """
    model = getattr(app.state, "forecast_model", None) or getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not ready to serve predictions.",
        )

    try:
        input_data = pd.DataFrame([payload.to_model_input_dict()])
        raw_prediction = model.predict(input_data)[0]

        # Multi-output model returns an array-like of predictions [step_1, step_2, step_3]
        if hasattr(raw_prediction, "__len__") and len(raw_prediction) >= 2:
            step_1 = round(float(raw_prediction[0]), 4)
            step_2 = round(float(raw_prediction[1]), 4)
            step_3 = (
                round(float(raw_prediction[2]), 4)
                if (payload.include_step_3 and len(raw_prediction) >= 3)
                else None
            )
        else:
            # Single-output fallback guard
            val = round(float(raw_prediction), 4)
            step_1, step_2, step_3 = val, val, val if payload.include_step_3 else None

        return ForecastResponse(
            forecast_step_1=step_1,
            forecast_step_2=step_2,
            forecast_step_3=step_3,
            predicted_total_orders=step_1,
            status="success",
            observation_horizon="Next observation steps (t+1, t+2, t+3)",
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting failed due to internal error: {exc}",
        )


# ---------------------------------------------------------------------------
# Prediction & Forecasting Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/forecast",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Forecast future observation steps (Step 1, Step 2, and optional Step 3)",
)
def forecast_future_demand(payload: OrderForecastRequest):
    """
    Accepts current daily operational metrics and produces future forecasts for:
    - Next Step 1 (t+1)
    - Next Step 2 (t+2)
    - Optional Next Step 3 (t+3)
    """
    return _execute_forecast(payload)


@app.post(
    "/predict",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Generate future demand forecasts (Backward-compatible /predict route)",
)
def predict_daily_demand(payload: OrderPredictionRequest):
    """
    Backward-compatible prediction route. Returns the multi-step future forecasts
    and provides `predicted_total_orders` as an alias for Next Step 1.
    """
    return _execute_forecast(payload)
