from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Dict
import math
import logging

# Initialize FastAPI app
app = FastAPI()

# Load models and feature names
try:
    loaded_model_weekly = joblib.load('final_model_weekly_sales.joblib')
    loaded_model_monthly = joblib.load('final_model_monthly_sales.joblib')
    loaded_model_discount = joblib.load('final_model_discount_percentage.joblib')
    loaded_feature_names = joblib.load('feature_names.joblib')
except Exception as e:
    raise HTTPException(status_code=500, detail="Error loading models or feature names")

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
        processed_data["data"]['Suggested Discount (%)'] = predictions_discount

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
                    "Suggested Discount (%)": sanitize_predictions([row["Suggested Discount (%)"]])[0]
                })

        # Group the results by product name and calculate the average prediction for each
        grouped_result = pd.DataFrame(result)
        grouped_result = grouped_result.groupby('Product Name').agg({
            'Predicted Next Week Sales': 'mean',
            'Predicted Next Month Sales': 'mean',
            'Suggested Discount (%)': 'mean'
        }).reset_index()

        # Round and convert to integers
        grouped_result['Predicted Next Week Sales'] = grouped_result['Predicted Next Week Sales'].round().astype(int)
        grouped_result['Predicted Next Month Sales'] = grouped_result['Predicted Next Month Sales'].round().astype(int)
        grouped_result['Suggested Discount (%)'] = grouped_result['Suggested Discount (%)'].round().astype(int)

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

    # Fill missing values (in case there are any other NaNs in columns)
    data.fillna(0, inplace=True)

    # Perform the analyses

    # 1. Correlation Analysis
    def generate_correlation_summary(correlation_matrix):
        correlation_summary = ""
        for col1 in correlation_matrix.columns:
            for col2 in correlation_matrix.columns:
                if col1 != col2:
                    correlation_value = correlation_matrix.loc[col1, col2]
                    if abs(correlation_value) > 0.7:
                        correlation_summary += f"Strong correlation between {col1} and {col2}: {correlation_value:.2f}\n"
                    elif abs(correlation_value) > 0.5:
                        correlation_summary += f"Moderate correlation between {col1} and {col2}: {correlation_value:.2f}\n"
                    elif abs(correlation_value) > 0.3:
                        correlation_summary += f"Weak correlation between {col1} and {col2}: {correlation_value:.2f}\n"
        return correlation_summary

    numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
    if not numeric_data.empty:
        correlation_summary = generate_correlation_summary(numeric_data.corr())
    else:
        correlation_summary = "No numeric data available for correlation analysis."
    
    # 2. Sales by Product Category (Total Sales and Average Sales)
    category_sales = data.groupby('Category').agg(
        Total_Sales=('Total Sales ($)', 'sum'),
        Average_Sales=('Total Sales ($)', 'mean')
    ).reset_index()

    # Round values to 2 decimal places
    category_sales['Total_Sales'] = category_sales['Total_Sales'].round(2)
    category_sales['Average_Sales'] = category_sales['Average_Sales'].round(2)

    # 3. Sales by Weather Condition (Total Sales and Average Sales)
    weather_sales = data.groupby('Weather Condition').agg(
        Total_Sales=('Total Sales ($)', 'sum'),
        Average_Sales=('Total Sales ($)', 'mean')
    ).reset_index()

    # Round values to 2 decimal places
    weather_sales['Total_Sales'] = weather_sales['Total_Sales'].round(2)
    weather_sales['Average_Sales'] = weather_sales['Average_Sales'].round(2)

    # 4. Monthly Sales Trend (Total Sales per Month)
    # Extracting month name only, no year
    data['Month'] = data['Date'].dt.month_name()  # Month name (e.g., January, February)
    data['Year'] = data['Date'].dt.year  # Extract the year to group by it as well
    
    # Grouping by both Year and Month, and then calculating total sales
    monthly_sales_per_year = data.groupby(['Year', 'Month'])['Total Sales ($)'].sum().reset_index(name="Total Monthly Sales")
    
    # Calculate the average monthly sales across all years for each month
    monthly_avg_sales_over_year = monthly_sales_per_year.groupby('Month')['Total Monthly Sales'].mean().reset_index(name="Average Monthly Sales")
    
    # Calculate the total sales across all years for each month
    monthly_total_sales = monthly_sales_per_year.groupby('Month')['Total Monthly Sales'].sum().reset_index(name="Total Sales ($)")
    
    # Merge the total and average sales by Month
    monthly_sales_over_year = pd.merge(monthly_avg_sales_over_year, monthly_total_sales, on='Month')

    # Round values to 2 decimal places
    monthly_sales_over_year['Average Monthly Sales'] = monthly_sales_over_year['Average Monthly Sales'].round(2)
    monthly_sales_over_year['Total Sales ($)'] = monthly_sales_over_year['Total Sales ($)'].round(2)

    # Sort months in calendar order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sales_over_year['Month'] = pd.Categorical(monthly_sales_over_year['Month'], categories=month_order, ordered=True)
    monthly_sales_over_year = monthly_sales_over_year.sort_values('Month')  # Sort by month

    # 5. Product Sales (Total and Average Sales by Product)
    product_sales = data.groupby('Product Name').agg(
        Total_Sales=('Total Sales ($)', 'sum'),
        Average_Sales=('Total Sales ($)', 'mean')
    ).reset_index()

    # Round values to 2 decimal places
    product_sales['Total_Sales'] = product_sales['Total_Sales'].round(2)
    product_sales['Average_Sales'] = product_sales['Average_Sales'].round(2)

    # 6. Sales by Season (Total Sales and Average Sales by Season)
    season_sales = data.groupby('Season').agg(
        Total_Sales=('Total Sales ($)', 'sum'),
        Average_Sales=('Total Sales ($)', 'mean')
    ).reset_index()

    # Round values to 2 decimal places
    season_sales['Total_Sales'] = season_sales['Total_Sales'].round(2)
    season_sales['Average_Sales'] = season_sales['Average_Sales'].round(2)

    # 7. Sales by Day of Week (Total Sales and Average Sales by Day of Week)
    data['Day of Week'] = data['Date'].dt.day_name()  # Extracting the day of the week
    day_of_week_sales = data.groupby('Day of Week').agg(
        Total_Sales=('Total Sales ($)', 'sum'),
        Average_Sales=('Total Sales ($)', 'mean')
    ).reset_index()

    # Round values to 2 decimal places
    day_of_week_sales['Total_Sales'] = day_of_week_sales['Total_Sales'].round(2)
    day_of_week_sales['Average_Sales'] = day_of_week_sales['Average_Sales'].round(2)

    # Return the analysis results
    analysis_result = {
        "correlation_summary": correlation_summary,  # Updated to return the correlation summary
        "category_sales": category_sales.to_dict(orient='records'),
        "weather_sales": weather_sales.to_dict(orient='records'),
        "monthly_sales_over_year": monthly_sales_over_year.to_dict(orient='records'),  # Updated to include total and average sales
        "product_sales": product_sales.to_dict(orient='records'),  # Combined Total and Average Sales for Products
        "season_sales": season_sales.to_dict(orient='records'),
        "day_of_week_sales": day_of_week_sales.to_dict(orient='records')
    }

    return analysis_result
