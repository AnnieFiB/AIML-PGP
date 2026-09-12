
import os
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download

# ---------------------------------------------------------
# LOAD MODEL INFORMATION
# ---------------------------------------------------------
with open("models/model_output.txt", "r") as f:
    model_filename = f.read().strip()

with open("models/model_repo.txt", "r") as f:
    model_repo = f.read().strip()

# Hugging Face token from environment / Space secret
hf_token = os.getenv("HF_TOKEN_ML")

# ---------------------------------------------------------
# DOWNLOAD AND LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():

    model_path = hf_hub_download( repo_id=model_repo, filename=model_filename,repo_type="model", token=hf_token)

    return joblib.load(model_path)

model = load_model()

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("Tourism Package Prediction")
st.write( "Fill in the customer details below to predict whether they are likely to purchase a travel package.")

# ---------------------------------------------------------
# CUSTOMER INPUTS
# ---------------------------------------------------------
Age = st.slider("Age", 18,70, 30)
TypeofContact = st.selectbox( "Type of Contact",  ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)",0,100,15)

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
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.slider(
    "Number of Persons Visiting",
    1,
    5,
    2
)

NumberOfFollowups = st.slider(
    "Number of Follow-ups",
    1,
    10,
    3
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
    1,
    20,
    3
)

Passport = st.selectbox(
    "Has Passport?",
    ["Yes", "No"]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.selectbox(
    "Owns a Car?",
    ["Yes", "No"]
)

NumberOfChildrenVisiting = st.slider(
    "Number of Children Visiting",
    0,
    5,
    1
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
    value=30000.0
)


# ---------------------------------------------------------
# PREPARE INPUT DATA
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

    "Passport": (
        1 if Passport == "Yes" else 0
    ),

    "PitchSatisfactionScore": PitchSatisfactionScore,

    "OwnCar": (
        1 if OwnCar == "Yes" else 0
    ),

    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])


# ---------------------------------------------------------
# CLASSIFICATION THRESHOLD
# ---------------------------------------------------------

classification_threshold = 0.45


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button("Predict"):

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= classification_threshold
    )

    if prediction == 1:

        st.success(
            "Customer is likely to purchase "
            "the travel package."
        )

    else:

        st.warning(
            "Customer is unlikely to purchase "
            "the travel package."
        )

    st.write(
        f"Purchase Probability: "
        f"**{probability:.2%}**"
    )
