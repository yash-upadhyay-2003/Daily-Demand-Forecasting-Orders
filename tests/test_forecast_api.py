"""
Tests for Daily Demand Multi-Step Future Forecasting API and model logic.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import ForecastResponse, HealthResponse, OrderPredictionRequest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verifies that the root endpoint returns 200 and required links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "forecast_url" in data
    assert "predict_url" in data


def test_health_check(client):
    """Verifies application readiness and forecast model loading."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    validated = HealthResponse(**data)
    assert validated.status == "healthy"
    assert validated.model_loaded is True
    assert validated.forecast_model_ready is True


def test_forecast_multi_step_all_horizons(client):
    """
    Tests /forecast endpoint returning forecasts for:
      - Next Step 1 (t+1)
      - Next Step 2 (t+2)
      - Optional Next Step 3 (t+3)
    """
    payload = {
        "Week of the month (first week, second, third, fourth or fifth week": 1,
        "Day of the week (Monday to Friday)": 4,
        "Non-urgent order": 316.307,
        "Urgent order": 223.27,
        "Fiscal sector orders": 0.0,
        "Orders from the traffic controller sector": 65556,
        "Banking orders (1)": 44914,
        "Banking orders (2)": 188411,
        "Banking orders (3)": 14793,
        "include_step_3": True,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    validated = ForecastResponse(**data)
    assert validated.status == "success"
    assert validated.forecast_step_1 > 0
    assert validated.forecast_step_2 > 0
    assert validated.forecast_step_3 is not None
    assert validated.forecast_step_3 > 0
    # Backward compatible alias check
    assert validated.predicted_total_orders == validated.forecast_step_1
    assert "t+1" in validated.observation_horizon


def test_forecast_step_3_optional_toggle(client):
    """Verifies that when include_step_3=False, forecast_step_3 is None."""
    payload = {
        "Week of the month (first week, second, third, fourth or fifth week": 2,
        "Day of the week (Monday to Friday)": 3,
        "Non-urgent order": 120.0,
        "Urgent order": 95.0,
        "Fiscal sector orders": 10.0,
        "Orders from the traffic controller sector": 45000,
        "Banking orders (1)": 30000,
        "Banking orders (2)": 120000,
        "Banking orders (3)": 15000,
        "include_step_3": False,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    validated = ForecastResponse(**data)
    assert validated.forecast_step_1 > 0
    assert validated.forecast_step_2 > 0
    assert validated.forecast_step_3 is None


def test_predict_backward_compatibility(client):
    """Verifies that POST /predict returns multi-step forecasts with predicted_total_orders."""
    payload = {
        "Week of the month (first week, second, third, fourth or fifth week": 1,
        "Day of the week (Monday to Friday)": 4,
        "Non-urgent order": 316.307,
        "Urgent order": 223.27,
        "Fiscal sector orders": 0.0,
        "Orders from the traffic controller sector": 65556,
        "Banking orders (1)": 44914,
        "Banking orders (2)": 188411,
        "Banking orders (3)": 14793,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "forecast_step_1" in data
    assert "forecast_step_2" in data
    assert "forecast_step_3" in data
    assert "predicted_total_orders" in data
    assert data["predicted_total_orders"] == data["forecast_step_1"]


def test_validation_errors(client):
    """Verifies that invalid day of week (e.g. 7 or 1) returns 422 Unprocessable Entity."""
    invalid_payload = {
        "Week of the month (first week, second, third, fourth or fifth week": 1,
        "Day of the week (Monday to Friday)": 7,  # Valid range is 2 to 6
        "Non-urgent order": 100.0,
        "Urgent order": 50.0,
        "Fiscal sector orders": 0.0,
        "Orders from the traffic controller sector": 10000,
        "Banking orders (1)": 10000,
        "Banking orders (2)": 10000,
        "Banking orders (3)": 10000,
    }
    response = client.post("/forecast", json=invalid_payload)
    assert response.status_code == 422


def test_no_order_type_leakage_in_schema():
    """Verifies that Order Type A, B, and C are NOT accepted as input features in request schema."""
    field_names = list(OrderPredictionRequest.model_fields.keys())
    assert "order_type_a" not in field_names
    assert "order_type_b" not in field_names
    assert "order_type_c" not in field_names
