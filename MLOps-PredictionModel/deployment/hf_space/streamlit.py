
import os

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tourism Wellness Package Prediction",
    page_icon="🌍"
    # ,layout="wide"
)

# ---------------------------------------------------------
# API CONFIGURATION
# ---------------------------------------------------------
API_URL = os.getenv( "PREDICTION_API_URL")

if not API_URL:
    st.error( "Prediction API URL is not configured." )
    st.stop()


API_URL = API_URL.rstrip("/")

PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"

# ---------------------------------------------------------
# API FUNCTIONS
# ---------------------------------------------------------
def get_model_info():
    """Retrieve information about the deployed model."""

    try:

        response = requests.get(HEALTH_URL, timeout=20 )
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None


def get_prediction(payload):
    """Send customer data to FastAPI."""

    response = requests.post(   PREDICT_URL, json=payload, timeout=60)
    response.raise_for_status()

    return response.json()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🌍 Tourism Prediction")
page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Information"
    ]
)

# =========================================================
# PREDICTION PAGE
# =========================================================
if page == "Prediction":

    st.title("🌍 Tourism Wellness Package Prediction" )
    st.caption("Enter customer information to estimate the likelihood of purchasing the tourism package.")

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER PROFILE
    # -----------------------------------------------------
    st.subheader(  "Customer Profile" )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input( "Age",   min_value=18, max_value=100, value=35 )
        gender = st.selectbox(  "Gender", ["Male", "Female"])
        marital_status = st.selectbox( "Marital Status",  [ "Single", "Married",  "Divorced","Unmarried"] )

    with col2:
        occupation = st.selectbox(  "Occupation", [ "Salaried",   "Small Business",  "Large Business",  "Free Lancer" ] )
        designation = st.selectbox( "Designation", [ "Executive", "Manager", "Senior Manager",  "AVP", "VP" ] )
        monthly_income = st.number_input( "Monthly Income",   min_value=0.0, value=25000.0, step=1000.0 )

    st.divider()

    # -----------------------------------------------------
    # TRAVEL & HOUSEHOLD
    # -----------------------------------------------------
    st.subheader(  "Travel & Household")

    col1, col2 = st.columns(2)

    with col1:
        city_tier = st.selectbox("City Tier",[1, 2, 3])
        number_trips = st.number_input("Number of Trips", min_value=0,  max_value=50, value=2  )
        passport_label = st.selectbox( "Passport", ["No", "Yes"  ] )
        passport = (  1 if passport_label == "Yes" else 0 )

    with col2:
        persons_visiting = st.number_input("Number of Persons Visiting", min_value=1,  max_value=20, value=2  )
        children_visiting = st.number_input("Number of Children Visiting",  min_value=0, max_value=10,  value=0 )
        own_car_label = st.selectbox( "Own Car", ["No", "Yes" ] )
        own_car = (  1 if own_car_label == "Yes" else 0 )

    st.divider()

    # -----------------------------------------------------
    # SALES & PACKAGE
    # -----------------------------------------------------
    st.subheader( "Sales & Package Information")

    col1, col2 = st.columns(2)

    with col1:
        contact_type = st.selectbox( "Type of Contact",  [ "Self Enquiry",  "Company Invited"  ] )
        product_pitched = st.selectbox( "Product Pitched", [ "Basic","Standard","Deluxe","Super Deluxe", "King"   ]   )
        duration_pitch = st.number_input(  "Duration of Pitch (minutes)",  min_value=0,max_value=120, value=15 )

    with col2:
        followups = st.number_input( "Number of Follow-ups", min_value=0, max_value=20,  value=3  )
        satisfaction = st.slider( "Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        property_star = st.slider("Preferred Property Star", min_value=1, max_value=5,value=3 )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------
    if st.button(  "Predict Purchase",  type="primary", use_container_width=True ):
        payload = {
            "Age": age,
            "TypeofContact": contact_type,
            "CityTier": city_tier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons_visiting,
            "PreferredPropertyStar": property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_trips,
            "Passport": passport,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
            "PitchSatisfactionScore": satisfaction,
            "ProductPitched": product_pitched,
            "NumberOfFollowups": followups,
            "DurationOfPitch": duration_pitch
        }

        try:
            with st.spinner(  "Generating prediction..." ):
                result = get_prediction(  payload )

            probability = result["purchase_probability"  ]

            if result["prediction"] == 1:
                st.success("Likely to Purchase" )
                st.write( "The customer is predicted to be a potential buyer of the tourism package." )
            else:
                st.warning("Unlikely to Purchase")
                st.write("The customer is currently predicted to have a lower likelihood of purchasing.")

            col1, col2 = st.columns(2)

            col1.metric( "Prediction", result["prediction_label"]  )
            col2.metric("Purchase Probability", f"{probability:.1%}")

            st.progress(  min( max(probability, 0.0), 1.0   ) )

        except requests.RequestException as exc:
            st.error(f"Prediction API unavailable: {exc}"  )

# =========================================================
# MODEL INFORMATION PAGE
# =========================================================
else:
    st.title("Model Information")
    st.caption("Information about the machine-learning model used by the prediction application.")
    st.divider()

    info = get_model_info()

    if info:
        col1, col2, col3 = st.columns(3)
        col1.metric( "Input Features", info.get( "input_features",18 ) )
        col2.metric( "Classification Threshold","45%")
        col3.metric( "Model Type", info.get("model_type","N/A") )

        st.subheader( "Deployed Model" )
        st.write( "**Model Repository:**",   info.get( "model_repo", "N/A") )
        st.write( "**Model File:**", info.get(  "model_file",  "N/A") )

        pipeline_steps = info.get( "pipeline_steps", [] )
        if pipeline_steps:
            st.subheader(  "Pipeline Components" )
            pipeline_df = pd.DataFrame({ "Pipeline Step":pipeline_steps  })
            st.dataframe(  pipeline_df, use_container_width=True,hide_index=True )

    else:
        st.warning(  "Unable to retrieve model information." )

    st.divider()

    st.subheader("MLOps Workflow")
    st.markdown(
        """
        **1. Dataset Registration**  
        Raw tourism data is stored in the Hugging Face Dataset Hub.

        **2. Data Preparation**  
        Data cleaning, preprocessing and stratified train/test splitting are performed automatically.

        **3. Model Training & Evaluation**  
        Multiple classification models are tuned and evaluated using cross-validation.

        **4. Model Selection**  
        The best-performing model is selected using F1 score.

        **5. Model Registry**  
        The final trained pipeline is stored in the Hugging Face Model Hub.

        **6. Deployment**  
        Hugging Face Space is updated with deployment files foor FastAPI + streamlit Interface to provide real-time predictions.
        """
    )
