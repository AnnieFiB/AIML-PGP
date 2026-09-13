
import os
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Tourism Wellness Package Prediction",
    page_icon="🌍",
    layout="wide"
)

# =========================================================
# MODEL CONFIGURATION
# =========================================================
with open("deployment/model_output.txt", "r") as f:
    MODEL_FILENAME = f.read().strip()

with open("deployment/model_repo.txt", "r") as f:
    MODEL_REPO = f.read().strip()


# Streamlit Cloud secret first, then local environment
try:
    HF_TOKEN = st.secrets["HF_TOKEN_ML"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN_ML")


CLASSIFICATION_THRESHOLD = 0.45

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    """
    Download and load the trained ML pipeline from Hugging Face Model Hub.
    """

    model_path = hf_hub_download(  repo_id=MODEL_REPO,  filename=MODEL_FILENAME,    repo_type="model",   token=HF_TOKEN )

    return joblib.load(model_path)

model = load_model()

# =========================================================
# MODEL INFORMATION
# =========================================================
# Identify final model automatically
if hasattr(model, "named_steps"):

    pipeline_steps = list(model.named_steps.keys())
    final_step_name = pipeline_steps[-1]
    final_model = model.named_steps[final_step_name]

    MODEL_TYPE = type(final_model).__name__

else:

    pipeline_steps = ["Model"]
    final_model = model

    MODEL_TYPE = type(model).__name__


INPUT_FEATURES = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🌍 Tourism Wellness Package")
page = st.sidebar.radio("Navigation", [ "Prediction", "Model Information"   ])
st.sidebar.divider()
st.sidebar.caption("MLOps Tourism Package Prediction")

# =========================================================
# PREDICTION PAGE
# =========================================================
if page == "Prediction":

    st.title("🌍 Tourism Wellness Package Prediction")
    st.write( "Enter the customer's information below to predict "  "their likelihood of purchasing the tourism package." )
    st.divider()

    # -----------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------

    with st.form("prediction_form"):

        # =================================================
        # CUSTOMER PROFILE
        # =================================================
        st.subheader("Customer Profile")

        col1, col2 = st.columns(2)

        with col1:
            Age = st.number_input(  "Age",min_value=18,  max_value=100, value=30)
            Gender = st.selectbox("Gender", ["Male", "Female"])
            MaritalStatus = st.selectbox( "Marital Status", [ "Married","Single","Divorced",  "Unmarried"  ] )

        with col2:
            Occupation = st.selectbox("Occupation",["Salaried", "Small Business", "Large Business", "Free Lancer"] )
            Designation = st.selectbox( "Designation",["Executive","Manager", "Senior Manager","AVP","VP","Other"])
            MonthlyIncome = st.number_input( "Monthly Income", min_value=0.0,value=30000.0,step=1000.0 )

        st.divider()

        # =================================================
        # TRAVEL & HOUSEHOLD
        # =================================================
        st.subheader("Travel & Household Information")

        col1, col2 = st.columns(2)

        with col1:
            CityTier = st.selectbox("City Tier",[1, 2, 3])
            NumberOfTrips = st.number_input("Number of Trips",min_value=0,value=3)
            Passport = st.selectbox("Has Passport?",[ "No","Yes"])

        with col2:
            NumberOfPersonVisiting = st.number_input("Number of Persons Visiting",  min_value=1, value=2)
            NumberOfChildrenVisiting = st.number_input( "Number of Children Visiting", min_value=0, value=1)
            OwnCar = st.selectbox("Owns a Car?",[ "No", "Yes"] )

        st.divider()

        # =================================================
        # SALES & PACKAGE INFORMATION
        # =================================================
        st.subheader("Sales & Package Information")

        col1, col2 = st.columns(2)

        with col1:
            TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry","Company Invited"])
            ProductPitched = st.selectbox( "Product Pitched",  ["Basic", "Standard","Deluxe","Super Deluxe","King"])
            DurationOfPitch = st.number_input( "Duration of Pitch (Minutes)", min_value=0, value=15 )

        with col2:
            NumberOfFollowups = st.number_input("Number of Follow-ups",min_value=0,value=3 )
            PitchSatisfactionScore = st.selectbox( "Pitch Satisfaction Score",[1, 2, 3, 4, 5])
            PreferredPropertyStar = st.selectbox("Preferred Property Star",[1, 2, 3, 4, 5] )

        st.write("")

        submitted = st.form_submit_button( "Predict Purchase Likelihood",use_container_width=True   )

    # =====================================================
    # PREDICTION
    # =====================================================
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

        probability = model.predict_proba( input_data )[0, 1]
        prediction = int(probability >= CLASSIFICATION_THRESHOLD)

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------
        st.divider()
        st.subheader("Prediction Result")
        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:
                st.success("Likely to Purchase" )
                st.write( "The customer is predicted to be a potential buyer of the tourism package." )
            else:
                st.warning("Unlikely to Purchase"  )
                st.write("The customer is currently predicted to have a lower likelihood of purchasing.")

        with result_col2:
            st.metric("Purchase Probability",f"{probability:.2%}" )
            st.progress( float(probability))

        st.caption(f"Classification threshold: "  f"{CLASSIFICATION_THRESHOLD:.0%}" )

# =========================================================
# MODEL INFORMATION PAGE
# =========================================================
elif page == "Model Information":

    st.title("Model Information")
    st.write("Information about the machine-learning model used by the prediction application.")
    st.divider()

    # -----------------------------------------------------
    # MODEL SUMMARY
    # -----------------------------------------------------
    st.subheader("Model Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric( "Input Features", len(INPUT_FEATURES) )

    with col2:
        st.metric( "Classification Threshold",f"{CLASSIFICATION_THRESHOLD:.0%}")

    with col3:
        st.metric( "Model Type", MODEL_TYPE )

    st.divider()

    # -----------------------------------------------------
    # DEPLOYED MODEL
    # -----------------------------------------------------
    st.subheader("Deployed Model")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model File**")
        st.code( MODEL_FILENAME,language=None)

    with col2:
        st.write("**Hugging Face Model Repository**")
        st.code(MODEL_REPO,language=None)

    # -----------------------------------------------------
    # PIPELINE
    # -----------------------------------------------------
        st.subheader("⚙️ Machine Learning Pipeline")
        st.write("The deployed object contains both preprocessing and the trained classifier, ensuring the same transformations are applied during training and inference." )

        if hasattr(model, "named_steps"):

            pipeline_df = pd.DataFrame({
                "Step": range(
                    1,
                    len(model.named_steps) + 1
                ),

                "Pipeline Component": [
                    name
                    for name in model.named_steps.keys()
                ],

                "Type": [
                    type(component).__name__
                    for component in model.named_steps.values()
                ]

            })

            st.dataframe(  pipeline_df,use_container_width=True,    hide_index=True)

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------
    st.subheader("Model Input Features")
    st.write("The application collects **18 original customer features**. Categorical variables are subsequently transformed by the preprocessing pipeline before prediction." )

    feature_col1, feature_col2 = st.columns(2)
    midpoint = len(INPUT_FEATURES) // 2

    with feature_col1:
        for feature in INPUT_FEATURES[:midpoint]:
            st.write(f"• {feature}"  )

    with feature_col2:
        for feature in INPUT_FEATURES[midpoint:]:
            st.write(f"• {feature}" )

    st.divider()

    # -----------------------------------------------------
    # MODEL WORKFLOW
    # -----------------------------------------------------
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
        Streamlit Community Cloud downloads the registered model and provides real-time predictions.
        """
    )

    # -----------------------------------------------------
    # PREDICTION LOGIC
    # -----------------------------------------------------
    with st.expander("How is the prediction calculated?"):
        st.write(
            """
            The model generates the probability that a customer will purchase the tourism package.

            A probability greater than or equal to the classification threshold is classified as **Likely to Purchase**.
            """
        )

        st.code(
            """
            probability = model.predict_proba(customer_data)[0, 1]
            prediction = int(probability >= 0.45 )
            """,
            language="python"
        )
