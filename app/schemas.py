"""
Request and response schemas for the Daily Demand Forecasting API.
Supports multi-step future forecasting for Next Step 1, Next Step 2, and optional Next Step 3.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class OrderPredictionRequest(BaseModel):
    """
    Input features expected by the daily demand forecasting pipeline.
    Field aliases match the exact feature names used in the model pipeline.
    """

    # Categorical features
    week_of_month: int = Field(
        ...,
        alias="Week of the month (first week, second, third, fourth or fifth week",
        description="Week number in the month (valid range: 1 to 5)",
        ge=1,
        le=5,
    )
    day_of_week: int = Field(
        ...,
        alias="Day of the week (Monday to Friday)",
        description="Day encoded as integer: Monday=2, Tuesday=3, Wednesday=4, Thursday=5, Friday=6",
        ge=2,
        le=6,
    )

    # Numerical features
    non_urgent_order: float = Field(
        ...,
        alias="Non-urgent order",
        description="Non-urgent order quantity metric",
        ge=0.0,
    )
    urgent_order: float = Field(
        ...,
        alias="Urgent order",
        description="Urgent order quantity metric",
        ge=0.0,
    )
    fiscal_sector_orders: float = Field(
        ...,
        alias="Fiscal sector orders",
        description="Order volume from fiscal sector",
        ge=0.0,
    )
    traffic_controller_orders: float = Field(
        ...,
        alias="Orders from the traffic controller sector",
        description="Order volume from traffic controller sector",
        ge=0.0,
    )
    banking_orders_1: float = Field(
        ...,
        alias="Banking orders (1)",
        description="Order volume from banking sector 1",
        ge=0.0,
    )
    banking_orders_2: float = Field(
        ...,
        alias="Banking orders (2)",
        description="Order volume from banking sector 2",
        ge=0.0,
    )
    banking_orders_3: float = Field(
        ...,
        alias="Banking orders (3)",
        description="Order volume from banking sector 3",
        ge=0.0,
    )

    # Optional forecasting configurations
    include_step_3: bool = Field(
        default=True,
        description="Whether to include optional Next Step 3 forecast in the response",
    )
    current_total_orders: Optional[float] = Field(
        default=None,
        alias="Current observation total orders",
        description="Optional current observation total orders for historical tracking",
        ge=0.0,
    )

    model_config = {
        # Allows creating instances using either the field name or the alias
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
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
        },
    }

    def to_model_input_dict(self) -> Dict[str, Any]:
        """
        Converts the request payload into a dictionary keyed by the exact feature
        names expected by the scikit-learn preprocessing ColumnTransformer.
        """
        return {
            "Week of the month (first week, second, third, fourth or fifth week": self.week_of_month,
            "Day of the week (Monday to Friday)": self.day_of_week,
            "Non-urgent order": self.non_urgent_order,
            "Urgent order": self.urgent_order,
            "Fiscal sector orders": self.fiscal_sector_orders,
            "Orders from the traffic controller sector": self.traffic_controller_orders,
            "Banking orders (1)": self.banking_orders_1,
            "Banking orders (2)": self.banking_orders_2,
            "Banking orders (3)": self.banking_orders_3,
        }


# Alias for explicit naming
OrderForecastRequest = OrderPredictionRequest


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class ForecastResponse(BaseModel):
    """
    Prediction output containing forecasts for future observation steps:
    Next Step 1 (t+1), Next Step 2 (t+2), and optional Next Step 3 (t+3).
    """

    forecast_step_1: float = Field(
        ...,
        description="Forecasted total orders for Next Step 1 (observation step t+1)",
    )
    forecast_step_2: float = Field(
        ...,
        description="Forecasted total orders for Next Step 2 (observation step t+2)",
    )
    forecast_step_3: Optional[float] = Field(
        default=None,
        description="Optional forecasted total orders for Next Step 3 (observation step t+3)",
    )
    predicted_total_orders: float = Field(
        ...,
        description="Backward-compatible alias returning the Next Step 1 forecast",
    )
    status: str = Field(
        default="success",
        description="Status flag indicating successful forecast",
    )
    observation_horizon: str = Field(
        default="Next observation steps (t+1, t+2, t+3)",
        description="Horizon description confirming sequential observation steps (no invented calendar dates)",
    )


# Backward-compatible alias
PredictionResponse = ForecastResponse


class HealthResponse(BaseModel):
    """Health check response confirming application and model readiness."""
    status: str = Field(..., description="Application health status")
    model_loaded: bool = Field(..., description="True if an ML model is ready for predictions")
    forecast_model_ready: bool = Field(
        default=True,
        description="True if the multi-step future forecasting pipeline is loaded",
    )
