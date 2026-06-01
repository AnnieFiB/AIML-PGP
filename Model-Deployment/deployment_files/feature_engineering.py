
from datetime import datetime

def create_features(X):
    X = X.copy()

    current_year = datetime.now().year

    X["Store_Age"] = current_year - X["Store_Establishment_Year"]
    X["Product_Category_Code"] = X["Product_Id"].str[:2]

    X = X.drop(
        columns=[
            "Product_Id",
            "Store_Id",
            "Store_Establishment_Year"
        ],
        errors="ignore"
    )

    return X
