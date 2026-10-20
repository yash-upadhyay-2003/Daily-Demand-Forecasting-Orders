"""
FastAPI application for Daily Demand Forecasting.
Serves predictions using the pre-trained regression pipeline.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

# Scikit-learn compatibility shims:
# The model was saved with scikit-learn 1.6.1. Newer scikit-learn releases
# refactored internal helper classes and attributes. These shims ensure the
# saved model unpickles seamlessly across scikit-learn versions without retraining.
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

import sklearn.linear_model as _lm
if not hasattr(_lm.LinearRegression, "tol"):
    _lm.LinearRegression.tol = None

from app.schemas import (
    HealthResponse,
    OrderPredictionRequest,
    PredictionResponse,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Locate the model artifact relative to the project root for portable deployment
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "demand_regression_modelAA.joblib"


# ---------------------------------------------------------------------------
# Model Lifecycle (Loaded once at startup)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the ML model once when the app starts,
    and release resources when the app shuts down.
    """
    if not MODEL_PATH.exists():
        app.state.model = None
        print(f"[ERROR] Model artifact not found at: {MODEL_PATH}")
    else:
        try:
            app.state.model = joblib.load(MODEL_PATH)
            print(f"[INFO] Successfully loaded model pipeline from: {MODEL_PATH}")
        except Exception as exc:
            app.state.model = None
            print(f"[ERROR] Failed to load model artifact: {exc}")

    yield

    # Clean up on shutdown
    app.state.model = None


# Initialize FastAPI app with descriptive metadata
app = FastAPI(
    title="Daily Demand Forecasting API",
    description="REST API for predicting daily total orders based on operational and sector demand metrics.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """
    Root endpoint confirming that the API service is up and running.
    """
    return {
        "message": "Daily Demand Forecasting API is running",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Health check endpoint reporting application and model readiness.
    """
    is_ready = getattr(app.state, "model", None) is not None
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load.",
        )
    return HealthResponse(status="healthy", model_loaded=True)


# ---------------------------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict daily total orders",
)
def predict_daily_demand(payload: OrderPredictionRequest):
    """
    Accepts daily operational order features, passes them into the pre-trained
    scikit-learn pipeline (preprocessing + regression model), and returns the
    predicted total number of orders.
    """
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions.",
        )

    try:
        # Convert request features to a 1-row DataFrame with the exact column names
        # expected by the pipeline's ColumnTransformer
        input_data = pd.DataFrame([payload.to_model_input_dict()])

        # Run prediction through the pipeline (includes OneHotEncoder + StandardScaler)
        raw_prediction = model.predict(input_data)[0]

        # Ensure prediction is a native Python float and rounded for clarity
        prediction_value = round(float(raw_prediction), 4)

        return PredictionResponse(
            predicted_total_orders=prediction_value,
            status="success",
        )

    except HTTPException:
        raise
    except Exception:
        # Hide internal stack traces from the client while returning a clear error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal processing error.",
        )
