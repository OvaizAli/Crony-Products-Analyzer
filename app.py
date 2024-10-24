import streamlit as st
import pandas as pd
import joblib

# Load the models and feature names
loaded_model_weekly = joblib.load('final_model_weekly_sales.joblib')
loaded_model_monthly = joblib.load('final_model_monthly_sales.joblib')
loaded_model_discount = joblib.load('final_model_discount_percentage.joblib')
loaded_feature_names = joblib.load('feature_names.joblib')

# Title of the application
st.title("Crony Products Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)
    st.info("Data Preview")
    st.dataframe(data)
    
    # Descriptive Statistics
    st.success("Statistical Summary for Numerical Columns")
    st.dataframe(data.describe())
    
    # Trend Analysis
    data['Date'] = pd.to_datetime(data['Date'])

    # Day of Week Analysis
    st.success("Total Sales by Day of the Week")
    data['Day of Week'] = data['Date'].dt.day_name()
    sales_by_day_of_week = data.groupby('Day of Week')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)', ascending=False).reset_index()
    st.dataframe(sales_by_day_of_week[['Day of Week', 'Total Sales ($)']])

    # Weekly Analysis
    st.success("Total Sales by Week")
    data['Week'] = data['Date'].dt.isocalendar().week
    sales_by_week = data.groupby('Week')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)', ascending=False).reset_index()
    st.dataframe(sales_by_week[['Week', 'Total Sales ($)']])

    # Sales Performance by Category/Product
    st.success("Total Sales by Product Category")
    category_sales = data.groupby('Category')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)', ascending=False).reset_index()
    st.dataframe(category_sales[['Category', 'Total Sales ($)']])
    
    st.success("Top 5 Products by Sales")
    product_sales = data.groupby('Product Name')['Total Sales ($)'].sum().reset_index()
    top_products = product_sales.sort_values(by='Total Sales ($)', ascending=False).head(5)
    st.dataframe(top_products)
    
    # Profitability Analysis
    st.success("Profitability Analysis")
    data['Profit Margin'] = data['Net Sales Value ($)'] / data['Total Sales ($)']
    profit_analysis = data.groupby('Product Name')['Profit Margin'].mean().reset_index()
    profit_analysis = profit_analysis.sort_values(by='Profit Margin', ascending=False).head(5)
    st.dataframe(profit_analysis)

    # Weather and Event Analysis
    st.success("Sales by Weather Condition")
    weather_sales = data.groupby('Weather Condition')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)', ascending=False)
    st.dataframe(weather_sales)
    
    st.success("Sales on Holidays vs Regular Days")
    holiday_sales = data.groupby('Is Holiday')['Total Sales ($)'].sum().reset_index()
    st.dataframe(holiday_sales)
    
    # Correlation Analysis for Important Factors
    st.success("Analysis for Important Factors Influencing Sales")

    data['Day of Week'] = data['Date'].dt.dayofweek
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Is Weekend'] = data['Day of Week'].apply(lambda x: 1 if x >= 5 else 0)

    data['Weather Condition Encoded'] = pd.factorize(data['Weather Condition'])[0]
    data['Is Holiday Encoded'] = data['Is Holiday'].apply(lambda x: 1 if x == 'Yes' else 0)

    features = ['Day of Week', 'Week', 'Is Weekend', 'Weather Condition Encoded', 'Is Holiday Encoded', 'Discount (%)', 'Total Sales ($)']
    correlation_matrix = data[features].corr()
    sales_correlation = correlation_matrix[['Total Sales ($)']].sort_values(by='Total Sales ($)', ascending=False)

    explanation = ""
    for feature in sales_correlation.index:
        if feature != 'Total Sales ($)':
            correlation_value = sales_correlation.loc[feature, 'Total Sales ($)']
            if correlation_value > 0:
                explanation += (f"- **{feature}** has a **positive** correlation of {correlation_value:.2f} with total sales.\n")
            elif correlation_value < 0:
                explanation += (f"- **{feature}** has a **negative** correlation of {correlation_value:.2f} with total sales.\n")
            else:
                explanation += (f"- **{feature}** has a **neutral** correlation with total sales.\n")

    st.markdown(explanation)

    # Stock Management and Inventory Turnover Analysis
    st.success("Top 5 Products by Stock After Sale")
    stock_data = data.groupby('Product Name')['Stock After Sale'].mean().reset_index()
    top_stock = stock_data.sort_values(by='Stock After Sale', ascending=False).head(5)
    st.dataframe(top_stock)
    
    inventory_turnover = (data['Total Sales ($)'] / data['Stock After Sale']).mean()
    st.success(f"Inventory Turnover Rate (Total Sales / Average Stock After Sale) = {inventory_turnover:.2f}")

    # Discount and Promotion Impact
    st.success("Sales with and without Discounts")
    discount_sales = data.groupby('Discount (%)')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)')
    st.dataframe(discount_sales)

    # Prediction Section for all data
    st.header("Predicted Weekly, Monthly Sales, and Discount Percentage")

    # Prepare the input data for prediction
    new_data = data.copy()

    # One-hot encoding for categorical features
    new_data = pd.get_dummies(new_data, columns=['Weather Condition', 'Season', 'Category', 'Product Name'], drop_first=True)

    # Handle missing values to avoid prediction errors
    new_data.fillna(0, inplace=True)

    # Convert 'Month' column to numeric if necessary
    if 'Month' in new_data.columns:
        try:
            new_data['Month'] = pd.to_datetime(new_data['Month']).dt.month
        except Exception as e:
            st.error(f"Error converting 'Month' to numeric: {e}")

    try:
        # Ensure that new_data has the same columns as the trained models
        features_week = new_data.reindex(columns=loaded_feature_names['weekly'], fill_value=0)
        features_month = new_data.reindex(columns=loaded_feature_names['monthly'], fill_value=0)
        features_discount = new_data.reindex(columns=loaded_feature_names['discount'], fill_value=0)

        # Check for non-numeric data in the feature sets
        if not features_week.select_dtypes(include=['object']).empty:
            st.error(f"Non-numeric data found in weekly features: {features_week.select_dtypes(include=['object']).columns.tolist()}")
        if not features_month.select_dtypes(include=['object']).empty:
            st.error(f"Non-numeric data found in monthly features: {features_month.select_dtypes(include=['object']).columns.tolist()}")
        if not features_discount.select_dtypes(include=['object']).empty:
            st.error(f"Non-numeric data found in discount features: {features_discount.select_dtypes(include=['object']).columns.tolist()}")

        # Make sure all features are numeric before predictions
        if features_week.select_dtypes(include=['object']).empty and features_month.select_dtypes(include=['object']).empty and features_discount.select_dtypes(include=['object']).empty:
            # Make predictions
            predictions_weekly = loaded_model_weekly.predict(features_week)
            predictions_monthly = loaded_model_monthly.predict(features_month)
            predictions_discount = loaded_model_discount.predict(features_discount)

            # Add predictions to the original data
            data['Next Week Sales'] = predictions_weekly
            data['Next Month Sales'] = predictions_monthly
            data['Predicted Discount (%)'] = predictions_discount

            # Calculate the previous average sales for each product
            data['Current Avg Weekly Sales'] = data.groupby('Product Name')['Future_7_Day_Sales'].transform('mean')
            data['Current Avg Monthly Sales'] = data.groupby('Product Name')['Monthly Sales'].transform('mean')
            data['Current Avg Discount (%)'] = data.groupby('Product Name')['Discount (%)'].transform('mean')

            # Group the data by product and calculate the average predicted sales and previous average sales
            grouped_data = data.groupby('Product Name', as_index=False).agg({
                'Current Avg Weekly Sales': 'mean',
                'Next Week Sales': 'mean',
                'Current Avg Monthly Sales': 'mean',
                'Next Month Sales': 'mean',
                'Current Avg Discount (%)': 'mean',
                'Predicted Discount (%)': 'mean'
            })

            # Round the sales values to whole numbers
            grouped_data['Current Avg Weekly Sales'] = grouped_data['Current Avg Weekly Sales'].round()
            grouped_data['Current Avg Monthly Sales'] = grouped_data['Current Avg Monthly Sales'].round()
            grouped_data['Current Avg Discount (%)'] = grouped_data['Current Avg Discount (%)'].round()
            grouped_data['Next Week Sales'] = grouped_data['Next Week Sales'].round()
            grouped_data['Next Month Sales'] = grouped_data['Next Month Sales'].round()
            grouped_data['Predicted Discount (%)'] = grouped_data['Predicted Discount (%)'].round()

            # Display the comparison of previous average sales vs predicted sales
            st.success("Comparison of Current Average Weekly and Monthly Sales with Predicted Sales and Discount (%)")
            st.dataframe(grouped_data)
        else:
            st.error("Error: Non-numeric data found in features. Check the output for details.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
