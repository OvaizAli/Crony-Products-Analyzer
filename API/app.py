from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Dict
import math
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load models and feature names
loaded_model_weekly = joblib.load('final_model_weekly_sales.joblib')
loaded_model_monthly = joblib.load('final_model_monthly_sales.joblib')
loaded_model_discount = joblib.load('final_model_discount_percentage.joblib')
loaded_feature_names = joblib.load('feature_names.joblib')

# Define the response model for prediction
class PredictionResponse(BaseModel):
    predictions_df: List[Dict[str, str]]  # List of dictionaries, each containing string fields for product name

    class Config:
        anystr_strip_whitespace = True

# Helper function to sanitize predictions
def sanitize_predictions(predictions):
    """Sanitize predictions to avoid out-of-range or NaN values."""
    return [0 if math.isnan(pred) or math.isinf(pred) else round(pred) for pred in predictions]

# Preprocess data function
def preprocess_data(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Perform necessary data preprocessing
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    if data['Date'].isna().any():
        raise HTTPException(status_code=400, detail="Invalid date found in data")

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

# Predict function
@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
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
        predictions_weekly = loaded_model_weekly.predict(processed_data["weekly"])
        predictions_monthly = loaded_model_monthly.predict(processed_data["monthly"])
        predictions_discount = loaded_model_discount.predict(processed_data["discount"])

        # Sanitize predictions to avoid NaN or Inf values
        predictions_weekly = sanitize_predictions(predictions_weekly)
        predictions_monthly = sanitize_predictions(predictions_monthly)
        predictions_discount = sanitize_predictions(predictions_discount)

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
                    "Product Name": product_name,
                    "Predicted Next Week Sales": sanitize_predictions([row["Predicted Next Week Sales"]])[0],
                    "Predicted Next Month Sales": sanitize_predictions([row["Predicted Next Month Sales"]])[0],
                    "Predicted Discount (%)": sanitize_predictions([row["Predicted Discount (%)"]])[0]
                })

        # Group the results by product name and calculate the average prediction for each
        grouped_result = pd.DataFrame(result)
        grouped_result = grouped_result.groupby('Product Name').agg({
            'Predicted Next Week Sales': 'mean',
            'Predicted Next Month Sales': 'mean',
            'Predicted Discount (%)': 'mean'
        }).reset_index()

        # Round and convert to integers
        grouped_result['Predicted Next Week Sales'] = grouped_result['Predicted Next Week Sales'].round().astype(int)
        grouped_result['Predicted Next Month Sales'] = grouped_result['Predicted Next Month Sales'].round().astype(int)
        grouped_result['Predicted Discount (%)'] = grouped_result['Predicted Discount (%)'].round().astype(int)

        # Convert the result to a list of dictionaries for the response
        result_dict = grouped_result.to_dict(orient='records')

        return {"predictions_df": result_dict}

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed due to an error")

# Data Analysis Endpoint
@app.post("/data-analysis/")
async def data_analysis(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
    except Exception as e:
        logging.error(f"File reading failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Check for required columns
    required_columns = ['Date', 'Total Sales ($)', 'Product Name', 'Weather Condition', 'Is Holiday', 'Season', 'Category']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_columns)}")

    # Ensure 'Date' column is in datetime format and handle any errors
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Drop rows where 'Date' could not be converted
    data = data.dropna(subset=['Date'])

    # Perform simple analysis (e.g., category breakdowns)
    data.fillna(0, inplace=True)

    # Example analysis: Sales by product category
    category_sales = data.groupby('Category')['Total Sales ($)'].sum().reset_index()

    # Example analysis: Sales by weather condition
    weather_sales = data.groupby('Weather Condition')['Total Sales ($)'].sum().reset_index()

    # Example analysis: Sales trend over time (e.g., monthly total sales)
    monthly_sales = data.groupby(data['Date'].dt.to_period('M'))['Total Sales ($)'].sum().reset_index(name="Monthly Sales")
    monthly_sales['Date'] = monthly_sales['Date'].astype(str)  # Convert to string for cleaner response

    # Example analysis: Average sales by product
    avg_product_sales = data.groupby('Product Name')['Total Sales ($)'].mean().reset_index(name="Average Sales")

    # Return the analysis results
    analysis_result = {
        "category_sales": category_sales.to_dict(orient='records'),
        "weather_sales": weather_sales.to_dict(orient='records'),
        "monthly_sales": monthly_sales.to_dict(orient='records'),
        "avg_product_sales": avg_product_sales.to_dict(orient='records')
    }

    return analysis_result
