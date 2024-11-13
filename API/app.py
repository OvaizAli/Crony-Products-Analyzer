from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Dict
import io

# Initialize FastAPI app
app = FastAPI()

# Load models and feature names
loaded_model_weekly = joblib.load('final_model_weekly_sales.joblib')
loaded_model_monthly = joblib.load('final_model_monthly_sales.joblib')
loaded_model_discount = joblib.load('final_model_discount_percentage.joblib')
loaded_feature_names = joblib.load('feature_names.joblib')

# Define the response model for prediction
class PredictionResponse(BaseModel):
    predictions_df: List[Dict[str, str]]  # List of dictionaries, each containing string fields for product name

    class Config:
        # Ensure that all string fields are treated as strings
        anystr_strip_whitespace = True

def preprocess_data(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Perform necessary data preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day of Week'] = data['Date'].dt.dayofweek
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Is Weekend'] = data['Day of Week'].apply(lambda x: 1 if x >= 5 else 0)
    data['Weather Condition Encoded'] = pd.factorize(data['Weather Condition'])[0]
    data['Is Holiday Encoded'] = data['Is Holiday'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # One-hot encode necessary features
    data = pd.get_dummies(data, columns=['Weather Condition', 'Season', 'Category', 'Product Name'], drop_first=True)
    data.fillna(0, inplace=True)  # Fill missing values

    # Ensure all data is numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(0, inplace=True)  # Refill any NaNs with 0 after coercion

    # Extract required features for each prediction
    features_week = data.reindex(columns=loaded_feature_names['weekly'], fill_value=0)
    features_month = data.reindex(columns=loaded_feature_names['monthly'], fill_value=0)
    features_discount = data.reindex(columns=loaded_feature_names['discount'], fill_value=0)

    return {
        "weekly": features_week,
        "monthly": features_month,
        "discount": features_discount,
        "data": data
    }

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Load the uploaded CSV file into a DataFrame
    try:
        data = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Preprocess the data
    processed_data = preprocess_data(data)

    # Ensure that all data is numeric after preprocessing
    if not all([processed_data[key].select_dtypes(include=['object']).empty for key in processed_data]):
        raise HTTPException(status_code=400, detail="Non-numeric data found in features after preprocessing")

    try:
        # Make predictions
        predictions_weekly = loaded_model_weekly.predict(processed_data["weekly"]).round()
        predictions_monthly = loaded_model_monthly.predict(processed_data["monthly"]).round()
        predictions_discount = loaded_model_discount.predict(processed_data["discount"]).round()

        # Add predictions to the data
        processed_data["data"]['Predicted Next Week Sales'] = predictions_weekly
        processed_data["data"]['Predicted Next Month Sales'] = predictions_monthly
        processed_data["data"]['Predicted Discount (%)'] = predictions_discount

        # Initialize a list to hold the results
        result = []

        # Loop over each row in the data
        for idx, row in processed_data["data"].iterrows():
            # Find the product name that has a value of 1 (which indicates it's the relevant product)
            product_name_column = [col for col in row.index if col.startswith("Product Name_") and row[col] == 1]

            if product_name_column:
                # Extract the product name (remove 'Product Name_' prefix)
                product_name = product_name_column[0].replace("Product Name_", "")

                # Append the relevant data (product name and its predictions)
                result.append({
                    "Product Name": product_name,  # Product name as string
                    "Predicted Next Week Sales": str(row["Predicted Next Week Sales"]),
                    "Predicted Next Month Sales": str(row["Predicted Next Month Sales"]),
                    "Predicted Discount (%)": str(row["Predicted Discount (%)"])
                })

        return {"predictions_df": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed due to an error")
