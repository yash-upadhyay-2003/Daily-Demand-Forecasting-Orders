"""
Streamlit Web Frontend for Daily Demand Forecasting Orders (Regression).
Connects to the deployed FastAPI backend on Render.
"""

import requests
import streamlit as st

# Backend endpoint configuration
API_BASE_URL = "https://daily-demand-forecasting-orders.onrender.com"
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

st.set_page_config(
    page_title="Daily Demand Forecasting — Regression",
    page_icon="📦",
    layout="wide",
)

st.title("Daily Demand Forecasting Orders")
st.caption(
    "Machine Learning Regression Model to predict Total Daily Orders based on operational and sector indicators."
)

st.divider()

# Input Form
with st.form("demand_prediction_form"):
    st.subheader("1. Calendar & Schedule Factors")

    col1, col2 = st.columns(2)

    with col1:
        week = st.selectbox(
            "Week of the month",
            options=[1, 2, 3, 4, 5],
            index=0,
            format_func=lambda x: {
                1: "First week (Week 1)",
                2: "Second week (Week 2)",
                3: "Third week (Week 3)",
                4: "Fourth week (Week 4)",
                5: "Fifth week (Week 5)",
            }[x],
            help="Select the week of the month (1 to 5)",
        )

    with col2:
        day = st.selectbox(
            "Day of the week",
            options=[2, 3, 4, 5, 6],
            index=2,
            format_func=lambda x: {
                2: "Monday",
                3: "Tuesday",
                4: "Wednesday",
                5: "Thursday",
                6: "Friday",
            }[x],
            help="Select the working day of the week (Monday to Friday)",
        )

    st.divider()
    st.subheader("2. Order Volumes & Sector Demands")

    col_left, col_right = st.columns(2)

    with col_left:
        non_urgent = st.number_input(
            "Non-urgent orders",
            min_value=0.0,
            value=316.307,
            step=1.0,
            format="%.3f",
            help="Non-urgent order volume metric",
        )
        urgent = st.number_input(
            "Urgent orders",
            min_value=0.0,
            value=223.270,
            step=1.0,
            format="%.3f",
            help="Urgent order volume metric",
        )
        fiscal = st.number_input(
            "Fiscal sector orders",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.3f",
            help="Orders from fiscal sector",
        )
        traffic = st.number_input(
            "Orders from traffic controller sector",
            min_value=0,
            value=65556,
            step=100,
            help="Orders monitored by traffic controller sector",
        )

    with col_right:
        banking_1 = st.number_input(
            "Banking orders (1)",
            min_value=0,
            value=44914,
            step=100,
            help="Banking sector 1 order volume",
        )
        banking_2 = st.number_input(
            "Banking orders (2)",
            min_value=0,
            value=188411,
            step=100,
            help="Banking sector 2 order volume",
        )
        banking_3 = st.number_input(
            "Banking orders (3)",
            min_value=0,
            value=14793,
            step=100,
            help="Banking sector 3 order volume",
        )

    submitted = st.form_submit_button(
        "Predict Total Orders",
        type="primary",
        use_container_width=True,
    )

if submitted:
    # Exact JSON schema expected by the deployed regression model pipeline
    payload = {
        "Week of the month (first week, second, third, fourth or fifth week": int(week),
        "Day of the week (Monday to Friday)": int(day),
        "Non-urgent order": float(non_urgent),
        "Urgent order": float(urgent),
        "Fiscal sector orders": float(fiscal),
        "Orders from the traffic controller sector": float(traffic),
        "Banking orders (1)": float(banking_1),
        "Banking orders (2)": float(banking_2),
        "Banking orders (3)": float(banking_3),
    }

    try:
        with st.spinner("Calling Daily Demand Forecasting model..."):
            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=90,
            )

        if response.ok:
            result = response.json()

            st.divider()
            st.subheader("Prediction Result")

            if isinstance(result, dict) and "predicted_total_orders" in result:
                predicted_val = float(result["predicted_total_orders"])
                st.metric(
                    label="Predicted Total Orders",
                    value=f"{predicted_val:,.2f}",
                )
                st.success("Regression forecast completed successfully.")
            else:
                st.warning("Unexpected response structure from prediction API:")
                st.json(result)

        else:
            st.error(f"API Error (HTTP {response.status_code})")
            try:
                error_body = response.json()
                st.write("**Error Details:**", error_body.get("detail", error_body))
            except Exception:
                st.code(response.text)

    except requests.exceptions.Timeout:
        st.error(
            "The prediction request timed out. The backend on Render might be waking up from sleep. "
            "Please wait a few moments and try again."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the backend server. Please verify your internet connection or backend status."
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"An error occurred while communicating with the prediction API: {exc}")
    except ValueError as exc:
        st.error(f"Failed to parse server response as JSON: {exc}")
