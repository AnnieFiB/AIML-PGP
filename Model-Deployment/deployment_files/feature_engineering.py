
import pandas as pd

def create_features(df):
    df = df.copy()

    if "Store_Establishment_Year" in df.columns:
        current_year = pd.Timestamp.now().year
        df["Store_Age"] = current_year - df["Store_Establishment_Year"]
        df.drop(columns=["Store_Establishment_Year"], inplace=True)

    return df
