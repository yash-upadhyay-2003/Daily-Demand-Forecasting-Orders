"""
Training script for Multi-Step Daily Demand Forecasting Pipeline.
Trains a direct multi-step forecasting pipeline to predict demand for:
  - Next Step 1 (t+1)
  - Next Step 2 (t+2)
  - Next Step 3 (t+3)
without leaking target components (Order Type A/B/C) and respecting time-series ordering.
Saves the artifact to `demand_forecast_multistep_model.joblib`.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Daily Demand Forecasting Orders.csv"
OUTPUT_MODEL_PATH = BASE_DIR / "demand_forecast_multistep_model.joblib"

FEATURE_COLS = [
    "Week of the month (first week, second, third, fourth or fifth week",
    "Day of the week (Monday to Friday)",
    "Non-urgent order",
    "Urgent order",
    "Fiscal sector orders",
    "Orders from the traffic controller sector",
    "Banking orders (1)",
    "Banking orders (2)",
    "Banking orders (3)",
]

CATEGORICAL_COLS = [
    "Week of the month (first week, second, third, fourth or fifth week",
    "Day of the week (Monday to Friday)",
]

NUMERICAL_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

TARGET_COLS = ["forecast_step_1", "forecast_step_2", "forecast_step_3"]


def load_and_prepare_data(csv_path: Path):
    """Loads CSV, verifies no target leakage, and builds multi-step future targets."""
    df = pd.read_csv(csv_path, index_col=0)

    # Verify target sum leakage rule:
    # Order type A + B + C equals total orders, so they MUST be excluded.
    leak_check = (df["Order type A"] + df["Order type B"] + df["Order type C"] - df["Target (Total orders)"]).abs().max()
    print(f"[DATA] Verified Order Type A+B+C leakage check: max diff = {leak_check:.6f}")

    clean = df.drop(
        columns=["Unnamed: 0", "Order type A", "Order type B", "Order type C"],
        errors="ignore",
    ).copy()

    # Formulate future observation steps
    clean["forecast_step_1"] = clean["Target (Total orders)"].shift(-1)
    clean["forecast_step_2"] = clean["Target (Total orders)"].shift(-2)
    clean["forecast_step_3"] = clean["Target (Total orders)"].shift(-3)

    # Drop the trailing steps where future targets are unknown
    prepared = clean.dropna(subset=TARGET_COLS).copy()
    print(f"[DATA] Prepared dataset: {len(prepared)} complete observation records for 3-step forecasting.")
    return prepared


def train_and_evaluate(data: pd.DataFrame):
    """Evaluates using TimeSeriesSplit and trains final multi-step pipeline."""
    X = data[FEATURE_COLS]
    y = data[TARGET_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numerical", StandardScaler(), NUMERICAL_COLS),
        ]
    )

    regressor = MultiOutputRegressor(Ridge(alpha=10.0, positive=True))
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])

    # TimeSeries Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = {col: [] for col in TARGET_COLS}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)

        for i, col in enumerate(TARGET_COLS):
            fold_mae = mean_absolute_error(y_val[col], preds[:, i])
            cv_maes[col].append(fold_mae)

    print("\n--- TimeSeries Cross-Validation (5 Splits) ---")
    for col in TARGET_COLS:
        print(f"  {col}: Mean MAE = {np.mean(cv_maes[col]):.2f}")

    # Train on full prepared dataset for production deployment
    pipeline.fit(X, y)
    print("\n[INFO] Final multi-step forecasting pipeline successfully fitted on all available historical steps.")
    return pipeline


def main():
    print(f"Loading data from {DATA_PATH}...")
    data = load_and_prepare_data(DATA_PATH)
    pipeline = train_and_evaluate(data)

    print(f"\nSaving multi-step forecast model to: {OUTPUT_MODEL_PATH}")
    joblib.dump(pipeline, OUTPUT_MODEL_PATH)
    print("[SUCCESS] Multi-step forecasting model saved successfully!")


if __name__ == "__main__":
    main()
