
import os
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)


# ---------------------------------------------------------
# READ MODEL CONFIGURATION
# ---------------------------------------------------------

with open("deployment/model_output.txt", "r") as f:
    MODEL_FILENAME = f.read().strip()

with open("deployment/model_repo.txt", "r") as f:
    MODEL_REPO = f.read().strip()


# Streamlit Cloud secret first, then local environment fallback
try:
    HF_TOKEN = st.secrets["HF_TOKEN_ML"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN_ML")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Download and load the trained pipeline
    from Hugging Face Model Hub.
    """

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=HF_TOKEN
    )

    return joblib.load(model_path)


model = load_model()


# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------

st.title("✈️ Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "they are likely to purchase the tourism package."
)


# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        Age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        TypeofContact = st.selectbox(
            "Type of Contact",
            [
                "Self Enquiry",
                "Company Invited"
            ]
        )

        CityTier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        DurationOfPitch = st.number_input(
            "Duration of Pitch",
            min_value=0,
            value=15
        )

        Occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer"
            ]
        )

        Gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        NumberOfPersonVisiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            value=2
        )

        NumberOfFollowups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            value=3
        )

        ProductPitched = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Standard",
                "Deluxe",
                "Super Deluxe",
                "King"
            ]
        )


    with col2:

        PreferredPropertyStar = st.selectbox(
            "Preferred Property Star",
            [1, 2, 3, 4, 5]
        )

        MaritalStatus = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Single",
                "Divorced",
                "Unmarried"
            ]
        )

        NumberOfTrips = st.number_input(
            "Number of Trips",
            min_value=0,
            value=3
        )

        Passport = st.selectbox(
            "Has Passport?",
            [
                "No",
                "Yes"
            ]
        )

        PitchSatisfactionScore = st.selectbox(
            "Pitch Satisfaction Score",
            [1, 2, 3, 4, 5]
        )

        OwnCar = st.selectbox(
            "Owns a Car?",
            [
                "No",
                "Yes"
            ]
        )

        NumberOfChildrenVisiting = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            value=1
        )

        Designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )

        MonthlyIncome = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=30000.0
        )


    submitted = st.form_submit_button(
        "Predict"
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome
    }])


    # Use 0.50 unless you have validated a different threshold
    CLASSIFICATION_THRESHOLD = 0.50


    probability = model.predict_proba(
        input_data
    )[0, 1]


    prediction = int(
        probability >= CLASSIFICATION_THRESHOLD
    )


    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "Customer is likely to purchase the tourism package."
        )

    else:

        st.warning(
            "Customer is unlikely to purchase the tourism package."
        )


    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )
