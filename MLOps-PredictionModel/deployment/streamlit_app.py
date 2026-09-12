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

with open("model_output.txt", "r") as f:
    MODEL_FILENAME = f.read().strip()

with open("model_repo.txt", "r") as f:
    MODEL_REPO = f.read().strip()


# Hugging Face secret
HF_TOKEN = os.getenv("HF_TOKEN_ML")


# ---------------------------------------------------------
# DOWNLOAD AND LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Download the trained model pipeline from Hugging Face
    Model Hub and load it for prediction.
    """

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=HF_TOKEN
    )

    return joblib.load(model_path)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

st.title("✈️ Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "they are likely to purchase the tourism package."
)


# ---------------------------------------------------------
# CUSTOMER INPUTS
# ---------------------------------------------------------

Age = st.slider(
    "Age",
    min_value=18,
    max_value=70,
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

DurationOfPitch = st.slider(
    "Duration of Pitch (minutes)",
    min_value=0,
    max_value=100,
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

NumberOfPersonVisiting = st.slider(
    "Number of Persons Visiting",
    min_value=1,
    max_value=5,
    value=2
)

NumberOfFollowups = st.slider(
    "Number of Follow-ups",
    min_value=1,
    max_value=10,
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

NumberOfTrips = st.slider(
    "Number of Trips",
    min_value=1,
    max_value=20,
    value=3
)

Passport = st.selectbox(
    "Has Passport?",
    [
        "Yes",
        "No"
    ]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

OwnCar = st.selectbox(
    "Owns a Car?",
    [
        "Yes",
        "No"
    ]
)

NumberOfChildrenVisiting = st.slider(
    "Number of Children Visiting",
    min_value=0,
    max_value=5,
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
    min_value=1000.0,
    value=30000.0,
    step=1000.0
)


# ---------------------------------------------------------
# PREPARE MODEL INPUT
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CLASSIFICATION THRESHOLD
# ---------------------------------------------------------

CLASSIFICATION_THRESHOLD = 0.45


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button(
    "Predict",
    use_container_width=True
):

    try:

        probability = model.predict_proba(
            input_data
        )[0, 1]

        prediction = int(
            probability >= CLASSIFICATION_THRESHOLD
        )

        st.divider()

        if prediction == 1:

            st.success(
                "Customer is likely to purchase "
                "the tourism package."
            )

        else:

            st.warning(
                "Customer is unlikely to purchase "
                "the tourism package."
            )

        st.metric(
            "Purchase Probability",
            f"{probability:.2%}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

with st.expander("Model Information"):

    st.write(
        f"**Model:** {MODEL_FILENAME}"
    )

    st.write(
        f"**Repository:** {MODEL_REPO}"
    )

    st.write(
        f"**Classification Threshold:** "
        f"{CLASSIFICATION_THRESHOLD}"
    )
