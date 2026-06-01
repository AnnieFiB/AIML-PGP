
import streamlit as st
import requests
import json
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="SuperKart Sales Forecasting Workbench",
    page_icon="🛒",
    layout="centered"
)

# -----------------------------
# Paths and config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

BACKEND_URL = "https://omotayof-superkart-backend.hf.space/predict"

AUTHOR_NAME = "Anthony Omotayo"
GITHUB_URL = ""

# -----------------------------
# Load model metrics dynamically
# -----------------------------
metrics_path = BASE_DIR / "model_metrics.json"

if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    MODEL_NAME = metrics.get("model_name", "Model")
    MODEL_VERSION = metrics.get("model_version", "v1.0")
    TEST_RMSE = float(metrics.get("test_rmse", 0))
    TEST_MAE = float(metrics.get("test_mae", 0))
    TEST_R2 = float(metrics.get("test_r2", 0))
    TEST_ADJ_R2 = float(metrics.get("test_adj_r2", 0))
    TEST_MAPE = float(metrics.get("test_mape", 0))

else:
    MODEL_NAME = "Model"
    MODEL_VERSION = "v1.0"
    TEST_RMSE = 0.0
    TEST_MAE = 0.0
    TEST_R2 = 0.0
    TEST_ADJ_R2 = 0.0
    TEST_MAPE = 0.0

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        max-width: 1000px;
    }

    .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        min-height: 70px;
        background-color: white;
        margin-bottom: 8px;
    }

    .kpi-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.2;
        word-wrap: break-word;
    }

    .forecast-card {
        background-color: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-top: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    .forecast-value {
        font-size: 2rem;
        font-weight: 700;
        color: #14532d;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    .muted {
        color: #64748b;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
col1, col2 = st.columns([1, 10])

with col1:
    st.markdown(
        "<h1 style='margin-top:10px'>🛒</h1>",
        unsafe_allow_html=True
    )

with col2:
    st.title("SuperKart Sales Forecasting Workbench")
    st.caption(
        "AI-powered retail revenue forecasting for product and store planning"
    )

# -----------------------------
# Tabs
# -----------------------------
st.divider()

tab1, tab2 = st.tabs(["Forecast", "Model Information"])

with tab1:

    st.markdown("### Forecast Setup")

    product_types = [
        "Frozen Foods",
        "Dairy",
        "Canned",
        "Baking Goods",
        "Health and Hygiene",
        "Meat",
        "Snack Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Household",
        "Soft Drinks",
        "Breakfast",
        "Bread",
        "Starchy Foods",
        "Seafood",
        "Others"
    ]

    store_types = [
        "Departmental Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Food Mart"
    ]

    with st.form("forecast_form"):

        with st.expander("Product Details", expanded=True):

            col1, col2 = st.columns(2)

            with col1:
                Product_Id = st.text_input("Product ID", value="FD123")

                Product_Weight = st.number_input(
                    "Product Weight",
                    min_value=0.0,
                    value=12.66
                )

                Product_Sugar_Content = st.selectbox(
                    "Sugar Content",
                    ["Low Sugar", "Regular", "No Sugar"]
                )

            with col2:
                Product_Allocated_Area = st.number_input(
                    "Allocated Display Area",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.027
                )

                Product_Type = st.selectbox(
                    "Product Type",
                    product_types
                )

                Product_MRP = st.number_input(
                    "Product MRP",
                    min_value=0.0,
                    value=117.08
                )

        with st.expander("Store Details", expanded=True):

            col3, col4 = st.columns(2)

            with col3:
                Store_Id = st.text_input("Store ID", value="ST001")

                Store_Establishment_Year = st.number_input(
                    "Store Establishment Year",
                    min_value=1900,
                    max_value=2100,
                    value=2009
                )

            with col4:
                Store_Size = st.selectbox(
                    "Store Size",
                    ["Small", "Medium", "High"]
                )

                Store_Location_City_Type = st.selectbox(
                    "City Tier",
                    ["Tier 1", "Tier 2", "Tier 3"]
                )

                Store_Type = st.selectbox(
                    "Store Type",
                    store_types
                )

        submitted = st.form_submit_button(
            "Generate Forecast",
            use_container_width=True
        )

    if submitted:

        payload = {
            "Product_Id": Product_Id,
            "Product_Weight": Product_Weight,
            "Product_Sugar_Content": Product_Sugar_Content,
            "Product_Allocated_Area": Product_Allocated_Area,
            "Product_Type": Product_Type,
            "Product_MRP": Product_MRP,
            "Store_Id": Store_Id,
            "Store_Establishment_Year": Store_Establishment_Year,
            "Store_Size": Store_Size,
            "Store_Location_City_Type": Store_Location_City_Type,
            "Store_Type": Store_Type
        }

        try:
            with st.spinner("Generating forecast..."):
                response = requests.post(
                    BACKEND_URL,
                    json=payload,
                    timeout=30
                )

            if response.status_code == 200:

                prediction = response.json()["predicted_sales"]

                lower_bound = max(0, prediction - TEST_RMSE)
                upper_bound = prediction + TEST_RMSE

                if prediction >= 4000:
                    demand_level = "High"
                    recommendation = (
                        "Strong revenue potential. Consider increasing stock levels "
                        "and prioritising shelf visibility for this product-store combination."
                    )

                elif prediction >= 2500:
                    demand_level = "Medium"
                    recommendation = (
                        "Moderate revenue potential. Maintain standard inventory allocation "
                        "and monitor demand trends."
                    )

                else:
                    demand_level = "Low"
                    recommendation = (
                        "Lower revenue potential. Consider cautious stock allocation "
                        "and review pricing, placement, or promotional strategy."
                    )

                st.markdown(
                    f"""
                    <div class="forecast-card">
                        <div class="muted">Predicted Sales Revenue</div>
                        <div class="forecast-value">£{prediction:,.2f}</div>
                        <div class="muted">
                            Forecast Range: £{lower_bound:,.0f} – £{upper_bound:,.0f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                c1.metric("Demand Signal", demand_level)
                c2.metric("Forecast Error Benchmark", f"±£{TEST_RMSE:,.0f}")

                st.subheader("Business Insight")
                st.info(recommendation)

            else:
                st.error("Forecast request failed.")
                st.json(response.json())

        except Exception as e:
            st.error(f"API connection error: {e}")

with tab2:

    k1, k2, k3, k4 = st.columns(4)

    cards = [
        ("Model", MODEL_NAME.replace("Tuned ", "")),
        ("RMSE", f"{TEST_RMSE:,.0f}"),
        ("R²", f"{TEST_R2*100:.1f}%"),
        ("Version", MODEL_VERSION)
    ]

    for col, (title, value) in zip([k1, k2, k3, k4], cards):

        with col:

            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{title}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{TEST_MAE:,.1f}")
    c2.metric("Adjusted R²", f"{TEST_ADJ_R2* 100:.1f}%")
    c3.metric("MAPE", f"{TEST_MAPE * 100:.1f}%")


    st.divider()

    st.subheader("Forecast Drivers")

    st.markdown(
        """
        - Product MRP
        - Product Category
        - Store Type
        - Store Size
        - Store Age
        - Allocated Shelf Area
        - City Tier
        """
    )

    st.divider()

    st.subheader("Deployment Architecture")

    st.markdown(
        """
        **Frontend:** Streamlit on Hugging Face  
        **Backend:** Flask API in Docker on Hugging Face  
        **Model:** Scikit-Learn Pipeline  
        **Deployment:** Hugging Face Spaces
        """
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    f"Built by {AUTHOR_NAME} | {MODEL_NAME} {MODEL_VERSION}"
)

if GITHUB_URL:
    st.caption(f"GitHub: {GITHUB_URL}")
