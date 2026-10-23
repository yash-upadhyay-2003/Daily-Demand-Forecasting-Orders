"""
Streamlit Web Frontend for Daily Demand Future Forecasting (Multi-Step).
Connects to the FastAPI backend (local dev server or Render deployment).
Displays forecasts for:
  - Next Working-Day Forecast
  - Forecast: 2 Working Days Ahead
  - Forecast: 3 Working Days Ahead
Does not invent calendar dates, respecting the dataset's Monday-Friday observation records.
"""

import os
import pandas as pd
import requests
import streamlit as st

# Backend endpoint configuration
DEFAULT_LOCAL_URL = "http://127.0.0.1:8000"
DEFAULT_PROD_URL = "https://daily-demand-forecasting-orders.onrender.com"

# Check environment variable for production override
ENV_BACKEND_URL = os.getenv("BACKEND_API_URL")

# Auto-detect default: if running locally with active port 8000, use local server.
# Otherwise (e.g. deployed on Streamlit Cloud or local server offline), use Render production.
def _detect_initial_backend_url() -> str:
    if ENV_BACKEND_URL:
        return ENV_BACKEND_URL
    try:
        resp = requests.get(f"{DEFAULT_LOCAL_URL}/health", timeout=0.3)
        if resp.ok:
            return DEFAULT_LOCAL_URL
    except Exception:
        pass
    return DEFAULT_PROD_URL

INITIAL_DEFAULT_URL = _detect_initial_backend_url()

st.set_page_config(
    page_title="Daily Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Backend Selection & Session Persistence (Retained across browser refresh)
# ---------------------------------------------------------------------------

# Restore selected backend from query parameters or session state
if "backend_url" not in st.session_state:
    query_param_backend = st.query_params.get("backend")
    if query_param_backend:
        st.session_state["backend_url"] = query_param_backend
    else:
        st.session_state["backend_url"] = INITIAL_DEFAULT_URL

current_backend = st.session_state.get("backend_url", INITIAL_DEFAULT_URL)

st.sidebar.header("⚙️ Server Configuration")

# Determine radio default index
if current_backend == DEFAULT_LOCAL_URL:
    default_env_idx = 0
elif current_backend == DEFAULT_PROD_URL:
    default_env_idx = 1
else:
    default_env_idx = 2

env_choice = st.sidebar.radio(
    "Target Environment",
    options=["Local Development", "Render Production", "Custom URL"],
    index=default_env_idx,
    help="Select the backend server environment.",
    key="env_choice_radio",
)

if env_choice == "Local Development":
    active_backend_url = DEFAULT_LOCAL_URL
elif env_choice == "Render Production":
    active_backend_url = DEFAULT_PROD_URL
else:
    active_backend_url = st.sidebar.text_input(
        "Custom Backend Base URL",
        value=current_backend if current_backend not in (DEFAULT_LOCAL_URL, DEFAULT_PROD_URL) else "http://localhost:8000",
        help="Specify custom backend base URL.",
        key="custom_backend_input",
    )

# Persist to session state and sync with browser URL query parameter
st.session_state["backend_url"] = active_backend_url
st.query_params["backend"] = active_backend_url

api_base_url = active_backend_url.rstrip("/")
FORECAST_URL = f"{api_base_url}/forecast"
PREDICT_URL = f"{api_base_url}/predict"
HEALTH_URL = f"{api_base_url}/health"

# Live health status check
try:
    health_resp = requests.get(HEALTH_URL, timeout=2.5)
    if health_resp.ok:
        health_data = health_resp.json()
        if health_data.get("forecast_model_ready", False):
            st.sidebar.success(f"🟢 **Backend Online (Multi-Step Ready)**\n\n`{api_base_url}`")
        else:
            st.sidebar.warning(f"🟡 **Backend Online (Legacy Single-Step Model)**\n\n`{api_base_url}`")
    else:
        st.sidebar.error(f"🔴 **Backend Error (HTTP {health_resp.status_code})**\n\n`{api_base_url}`")
except Exception:
    st.sidebar.error(
        f"🔴 **Backend Offline**\n\n`{api_base_url}`\n\n"
        "Start local server in PowerShell:\n`python -m uvicorn app.main:app --port 8000`"
    )

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------

st.title("📦 Daily Demand Forecasting")
st.caption(
    "Multi-step machine learning forecasting for daily total orders across upcoming working days "
    "using operational and sector demand indicators."
)

st.info(
    "ℹ️ **Working-Day Notice:** The dataset records Monday–Friday operational observations without verified calendar dates. "
    "Forecasts project total orders across upcoming working days (**Next Working-Day Forecast**, "
    "**Forecast: 2 Working Days Ahead**, and optional **Forecast: 3 Working Days Ahead**) rather than synthetic calendar dates. "
    "Target-leaking components (Order Types A, B, C) are excluded.",
    icon="📅",
)

st.divider()

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------

with st.form("demand_forecast_form"):
    st.subheader("1. Current Observation & Schedule Indicators")
    col1, col2 = st.columns(2)

    with col1:
        week = st.selectbox(
            "Week of the month (Current Observation)",
            options=[1, 2, 3, 4, 5],
            index=0,
            format_func=lambda x: {
                1: "First week (Week 1)",
                2: "Second week (Week 2)",
                3: "Third week (Week 3)",
                4: "Fourth week (Week 4)",
                5: "Fifth week (Week 5)",
            }[x],
            help="Select the week of the month for the Current Observation (1 to 5)",
        )

    with col2:
        day = st.selectbox(
            "Day of the week (Current Observation)",
            options=[2, 3, 4, 5, 6],
            index=2,
            format_func=lambda x: {
                2: "Monday (Day 2)",
                3: "Tuesday (Day 3)",
                4: "Wednesday (Day 4)",
                5: "Thursday (Day 5)",
                6: "Friday (Day 6)",
            }[x],
            help="Select the working day of the week for the Current Observation (Monday to Friday)",
        )

    st.divider()
    st.subheader("2. Operational Order Volumes & Sector Demands")

    col_left, col_right = st.columns(2)

    with col_left:
        non_urgent = st.number_input(
            "Non-urgent orders",
            min_value=0.0,
            value=316.307,
            step=1.0,
            format="%.3f",
            help="Non-urgent order volume metric for Current Observation",
        )
        urgent = st.number_input(
            "Urgent orders",
            min_value=0.0,
            value=223.270,
            step=1.0,
            format="%.3f",
            help="Urgent order volume metric for Current Observation",
        )
        fiscal = st.number_input(
            "Fiscal sector orders",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.3f",
            help="Order volume from fiscal sector for Current Observation",
        )
        traffic = st.number_input(
            "Orders from traffic controller sector",
            min_value=0,
            value=65556,
            step=100,
            help="Order volume monitored by traffic controller sector",
        )

    with col_right:
        banking_1 = st.number_input(
            "Banking orders (1)",
            min_value=0,
            value=44914,
            step=100,
            help="Order volume from banking sector 1",
        )
        banking_2 = st.number_input(
            "Banking orders (2)",
            min_value=0,
            value=188411,
            step=100,
            help="Order volume from banking sector 2",
        )
        banking_3 = st.number_input(
            "Banking orders (3)",
            min_value=0,
            value=14793,
            step=100,
            help="Order volume from banking sector 3",
        )

    st.divider()
    st.subheader("3. Forecasting Horizon & Options")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        include_step_3 = st.checkbox(
            "Include Forecast: 3 Working Days Ahead (Optional)",
            value=True,
            help="Check to display forecasts for Next Working Day, 2 Working Days Ahead, and 3 Working Days Ahead.",
        )
    with col_opt2:
        current_total = st.number_input(
            "Current Observation Total Orders (Optional baseline)",
            min_value=0.0,
            value=539.577,
            step=1.0,
            format="%.3f",
            help="Optional actual total orders observed for Current Observation for baseline comparison",
        )

    submitted = st.form_submit_button(
        "Generate Future Demand Forecast",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Forecast Execution & Display
# ---------------------------------------------------------------------------

if submitted:
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
        "include_step_3": bool(include_step_3),
        "Current observation total orders": float(current_total) if current_total > 0 else None,
    }

    try:
        with st.spinner(f"Calling Daily Demand Future Forecasting model at `{api_base_url}`..."):
            target_url = FORECAST_URL
            try:
                response = requests.post(target_url, json=payload, timeout=90)
                if response.status_code == 404:
                    response = requests.post(PREDICT_URL, json=payload, timeout=90)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                raise
            except Exception:
                response = requests.post(PREDICT_URL, json=payload, timeout=90)

        if response.ok:
            result = response.json()

            # Extract step forecasts
            step_1 = result.get("forecast_step_1")
            step_2 = result.get("forecast_step_2")
            step_3 = result.get("forecast_step_3")

            # Check if backend validly returned multi-step forecast
            if step_1 is not None and step_2 is not None:
                step_1_val = float(step_1)
                step_2_val = float(step_2)
                step_3_val = float(step_3) if step_3 is not None else None

                st.divider()
                st.subheader("📊 Demand Forecast for Upcoming Working Days")

                # Display Metric Cards with business-friendly working-day terminology
                if include_step_3 and step_3_val is not None:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric(
                            label="Next Working-Day Forecast",
                            value=f"{step_1_val:,.2f}",
                            delta=f"{step_1_val - current_total:+.2f} vs Current Observation" if current_total > 0 else None,
                            help="Forecasted total orders for the next working day",
                        )
                    with col_m2:
                        st.metric(
                            label="Forecast: 2 Working Days Ahead",
                            value=f"{step_2_val:,.2f}",
                            delta=f"{step_2_val - step_1_val:+.2f} vs Next Working Day",
                            help="Forecasted total orders two working days ahead",
                        )
                    with col_m3:
                        st.metric(
                            label="Forecast: 3 Working Days Ahead",
                            value=f"{step_3_val:,.2f}",
                            delta=f"{step_3_val - step_2_val:+.2f} vs 2 Days Ahead",
                            help="Optional forecasted total orders three working days ahead",
                        )
                else:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric(
                            label="Next Working-Day Forecast",
                            value=f"{step_1_val:,.2f}",
                            delta=f"{step_1_val - current_total:+.2f} vs Current Observation" if current_total > 0 else None,
                            help="Forecasted total orders for the next working day",
                        )
                    with col_m2:
                        st.metric(
                            label="Forecast: 2 Working Days Ahead",
                            value=f"{step_2_val:,.2f}",
                            delta=f"{step_2_val - step_1_val:+.2f} vs Next Working Day",
                            help="Forecasted total orders two working days ahead",
                        )

                # Forecast Trajectory Chart
                # Sequentially numbered, concise labels guarantee strict left-to-right ordering and prevent truncation on mobile
                chart_timeline_labels = [
                    "1. Current Observation",
                    "2. Next Working Day",
                    "3. 2 Days Ahead",
                ]
                chart_orders = [float(current_total), step_1_val, step_2_val]
                full_descriptions = [
                    "Current Observation",
                    "Next Working-Day Forecast",
                    "Forecast: 2 Working Days Ahead",
                ]

                if include_step_3 and step_3_val is not None:
                    chart_timeline_labels.append("4. 3 Days Ahead")
                    chart_orders.append(step_3_val)
                    full_descriptions.append("Forecast: 3 Working Days Ahead")

                chart_df = pd.DataFrame(
                    {"Forecasted Total Orders": chart_orders},
                    index=chart_timeline_labels,
                )
                chart_df.index.name = "Timeline"
                st.write("**Demand Forecast Progression Across Working Days:**")
                st.line_chart(chart_df)

                # Detailed Breakdown Table
                breakdown_df = pd.DataFrame({
                    "Working-Day Horizon": full_descriptions,
                    "Timeline Sequence": chart_timeline_labels,
                    "Forecasted Total Orders": [f"{v:,.2f}" for v in chart_orders],
                })
                with st.expander("🔍 View Tabular Breakdown by Working Day"):
                    st.dataframe(
                        breakdown_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.success(
                    "Demand forecast for upcoming working days generated successfully. "
                    "Predictions map operational lead indicators to sequential working days without target leakage."
                )

            else:
                # Connected backend returned legacy single-step prediction
                st.divider()
                st.error(
                    f"⚠️ **Multi-Step Forecast Unavailable from `{api_base_url}`:**\n\n"
                    "The connected backend returned a legacy single-step prediction rather than multi-step future forecasts. "
                    "This occurs when the backend server is running the legacy single-day model (e.g. un-updated Render deployment).\n\n"
                    "**To resolve:**\n"
                    "1. Ensure your local backend is running: `python -m uvicorn app.main:app --port 8000`\n"
                    "2. Verify the sidebar server selection is set to **Local Development (`http://127.0.0.1:8000`)**."
                )
                with st.expander("View Raw Response from Backend"):
                    st.json(result)

        else:
            st.error(f"API Error from `{api_base_url}` (HTTP {response.status_code})")
            try:
                error_body = response.json()
                st.write("**Error Details:**", error_body.get("detail", error_body))
            except Exception:
                st.code(response.text)

    except requests.exceptions.Timeout:
        st.error(
            f"The forecast request to `{api_base_url}` timed out.\n\n"
            "If connecting to Render, the instance may be in cold sleep. Please wait a moment and retry. "
            "If testing locally, please verify your local FastAPI server is running."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ **Unable to connect to backend server at `{api_base_url}`.**\n\n"
            "**Troubleshooting:**\n"
            "- If running locally, start FastAPI with: `python -m uvicorn app.main:app --port 8000`\n"
            "- Check the sidebar **Server Configuration** to ensure the correct host and port are selected."
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"An error occurred while communicating with the forecast API at `{api_base_url}`: {exc}")
    except ValueError as exc:
        st.error(f"Failed to parse server response from `{api_base_url}` as JSON: {exc}")
