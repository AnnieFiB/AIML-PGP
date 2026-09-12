import os
import joblib
import gradio as gr
import pandas as pd

from huggingface_hub import hf_hub_download


# ---------------------------------------------------------
# READ MODEL CONFIGURATION
# ---------------------------------------------------------

with open("model_output.txt", "r") as f:
    MODEL_FILENAME = f.read().strip()

with open("model_repo.txt", "r") as f:
    MODEL_REPO = f.read().strip()


HF_TOKEN = os.getenv("HF_TOKEN_ML")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def load_model():
    """
    Download and load the trained model pipeline
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
# PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_purchase(
    Age,
    TypeofContact,
    CityTier,
    DurationOfPitch,
    Occupation,
    Gender,
    NumberOfPersonVisiting,
    NumberOfFollowups,
    ProductPitched,
    PreferredPropertyStar,
    MaritalStatus,
    NumberOfTrips,
    Passport,
    PitchSatisfactionScore,
    OwnCar,
    NumberOfChildrenVisiting,
    Designation,
    MonthlyIncome
):

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

    # Classification threshold
    classification_threshold = 0.45

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= classification_threshold
    )

    if prediction == 1:
        result = "Customer is likely to purchase the tourism package."
    else:
        result = "Customer is unlikely to purchase the tourism package."

    return result, f"{probability:.2%}"


# ---------------------------------------------------------
# GRADIO INTERFACE
# ---------------------------------------------------------

with gr.Blocks(
    title="Tourism Package Prediction"
) as demo:

    gr.Markdown(
        """
        # Tourism Package Prediction

        Enter the customer details below to predict whether
        they are likely to purchase the tourism package.
        """
    )

    with gr.Row():

        with gr.Column():

            Age = gr.Slider(
                minimum=18,
                maximum=70,
                value=30,
                step=1,
                label="Age"
            )

            TypeofContact = gr.Dropdown(
                choices=[
                    "Self Enquiry",
                    "Company Invited"
                ],
                value="Self Enquiry",
                label="Type of Contact"
            )

            CityTier = gr.Dropdown(
                choices=[1, 2, 3],
                value=1,
                label="City Tier"
            )

            DurationOfPitch = gr.Slider(
                minimum=0,
                maximum=100,
                value=15,
                step=1,
                label="Duration of Pitch (minutes)"
            )

            Occupation = gr.Dropdown(
                choices=[
                    "Salaried",
                    "Small Business",
                    "Large Business",
                    "Free Lancer"
                ],
                value="Salaried",
                label="Occupation"
            )

            Gender = gr.Dropdown(
                choices=[
                    "Male",
                    "Female"
                ],
                value="Male",
                label="Gender"
            )

            NumberOfPersonVisiting = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="Number of Persons Visiting"
            )

            NumberOfFollowups = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of Follow-ups"
            )

            ProductPitched = gr.Dropdown(
                choices=[
                    "Basic",
                    "Standard",
                    "Deluxe",
                    "Super Deluxe",
                    "King"
                ],
                value="Basic",
                label="Product Pitched"
            )


        with gr.Column():

            PreferredPropertyStar = gr.Dropdown(
                choices=[1, 2, 3, 4, 5],
                value=3,
                label="Preferred Property Star"
            )

            MaritalStatus = gr.Dropdown(
                choices=[
                    "Married",
                    "Single",
                    "Divorced",
                    "Unmarried"
                ],
                value="Married",
                label="Marital Status"
            )

            NumberOfTrips = gr.Slider(
                minimum=1,
                maximum=20,
                value=3,
                step=1,
                label="Number of Trips"
            )

            Passport = gr.Dropdown(
                choices=[
                    "Yes",
                    "No"
                ],
                value="No",
                label="Has Passport?"
            )

            PitchSatisfactionScore = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Pitch Satisfaction Score"
            )

            OwnCar = gr.Dropdown(
                choices=[
                    "Yes",
                    "No"
                ],
                value="No",
                label="Owns a Car?"
            )

            NumberOfChildrenVisiting = gr.Slider(
                minimum=0,
                maximum=5,
                value=1,
                step=1,
                label="Number of Children Visiting"
            )

            Designation = gr.Dropdown(
                choices=[
                    "Executive",
                    "Manager",
                    "Senior Manager",
                    "AVP",
                    "VP"
                ],
                value="Executive",
                label="Designation"
            )

            MonthlyIncome = gr.Number(
                value=30000,
                label="Monthly Income"
            )


    predict_button = gr.Button(
        "Predict"
    )


    prediction_output = gr.Textbox(
        label="Prediction"
    )

    probability_output = gr.Textbox(
        label="Purchase Probability"
    )


    predict_button.click(
        fn=predict_purchase,

        inputs=[
            Age,
            TypeofContact,
            CityTier,
            DurationOfPitch,
            Occupation,
            Gender,
            NumberOfPersonVisiting,
            NumberOfFollowups,
            ProductPitched,
            PreferredPropertyStar,
            MaritalStatus,
            NumberOfTrips,
            Passport,
            PitchSatisfactionScore,
            OwnCar,
            NumberOfChildrenVisiting,
            Designation,
            MonthlyIncome
        ],

        outputs=[
            prediction_output,
            probability_output
        ]
    )


# ---------------------------------------------------------
# START APP
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch()
