import streamlit as st
import pandas as pd
import joblib
from mlxtend.frequent_patterns import apriori, association_rules

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
    
    st.header("Data Analysis")
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

    # # Sales Performance by Category/Product
    # st.success("Total Sales by Product Category")
    # category_sales = data.groupby('Category')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)', ascending=False).reset_index()
    # st.dataframe(category_sales[['Category', 'Total Sales ($)']])
    
    # st.success("Top 5 Products by Sales")
    # product_sales = data.groupby('Product Name')['Total Sales ($)'].sum().reset_index()
    # top_products = product_sales.sort_values(by='Total Sales ($)', ascending=False).head(5)
    # st.dataframe(top_products)
    
    # # Profitability Analysis
    # st.success("Profitability Analysis")
    data['Profit Margin'] = data['Net Sales Value ($)'] / data['Total Sales ($)']
    # profit_analysis = data.groupby('Product Name')['Profit Margin'].mean().reset_index()
    # profit_analysis = profit_analysis.sort_values(by='Profit Margin', ascending=False).head(5)
    # st.dataframe(profit_analysis)

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
    st.success("Top 5 Products by their Current Stock")
    stock_data = data.groupby('Product Name')['Stock After Sale'].mean().reset_index()
    top_stock = stock_data.sort_values(by='Stock After Sale', ascending=False).head(5)
    st.dataframe(top_stock)
    
    inventory_turnover = (data['Total Sales ($)'] / data['Stock After Sale']).mean()
    st.success(f"Inventory Turnover Rate (Total Sales / Average Stock After Sale) = {inventory_turnover:.2f}")

    # Discount and Promotion Impact
    st.success("Sales with and without Discounts")
    discount_sales = data.groupby('Discount (%)')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)')
    st.dataframe(discount_sales)

    # Prepare data in one-hot encoded format for product combinations
    # Pivot the data to create a basket format (Transaction ID as rows, Products as columns)
    basket = data.groupby(['Transaction ID', 'Product Name']).size().unstack().fillna(0)

    # Convert values greater than 0 to 1 (binary format)
    product_basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Run apriori algorithm with a lower min_support to capture more item combinations
    frequent_items = apriori(product_basket, min_support=0.005, use_colnames=True)

    # Generate association rules with lift metric
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)

    # Display only rules with two or more items in the antecedents or consequents
    filtered_rules = rules[(rules['antecedents'].apply(len) >= 1) | (rules['consequents'].apply(len) >= 1)]

   # Display results in Streamlit
    st.success("Products Often Bought Together")

    # Check if there are no rules and display an appropriate message
    if filtered_rules.empty:
        st.warning("We couldn't find any products that are frequently bought together.")
    else:
        # Rename columns for better understanding
        filtered_rules.columns = [
            'Items Purchased Together',  # Represents the products that were bought together
            'Suggested Products',         # Represents products recommended based on past purchases
            'Popularity (%)',             # Indicates how often this product combination was bought
            'Likelihood (%)',             # The chance of buying the suggested products if the items purchased together are bought
            'Strength of Association'     # A measure of how closely related the products are in terms of purchasing
        ]
        
        # Display the filtered rules
        st.dataframe(filtered_rules[['Items Purchased Together', 'Suggested Products', 'Popularity (%)', 'Likelihood (%)', 'Strength of Association']])

    # Calculate Weekly and Monthly Growth as percentages
    data['Weekly Growth (%)'] = data['Total Sales ($)'].pct_change(periods=7) 
    data['Monthly Growth (%)'] = data['Total Sales ($)'].pct_change(periods=30) 

    # Group by Product Name for high growth items, taking the mean for each product
    high_growth_items = data.groupby('Product Name')[['Weekly Growth (%)', 'Monthly Growth (%)']].mean().reset_index()

    # Display the results
    st.success("Sales Velocity and Growth Trends")
    st.dataframe(high_growth_items.round(2))

    # Calculate Price and Quantity Changes by Product Name
    data['Price Change (%)'] = data.groupby('Product Name')['Sales Price per Unit ($)'].pct_change() * 100
    data['Quantity Change (%)'] = data.groupby('Product Name')['Quantity Sold'].pct_change() * 100

    # Calculate Price Sensitivity Ratio (Elasticity)
    data['Price Sensitivity Ratio'] = data['Quantity Change (%)'] / data['Price Change (%)']

    # Group by Product Name and calculate the mean values for each column
    price_sensitivity_analysis = data.groupby('Product Name')[['Price Change (%)', 'Quantity Change (%)', 'Price Sensitivity Ratio']].mean().reset_index()

    # Display the Price Sensitivity Analysis
    st.success("Price Sensitivity Analysis (Price and Demand Shifts)")
    st.dataframe(price_sensitivity_analysis.round(2))

    category_performance = data.groupby('Category').agg({
        'Quantity Sold': 'sum',
        'Total Sales ($)': 'sum',
        'Profit Margin': 'mean'
    }).sort_values('Total Sales ($)', ascending=False)
    st.success("Category-Based Performance Analysis")
    st.dataframe(category_performance)

    # Prediction Section for all data
    st.header("Predicted Weekly, Monthly Sales, and Recommended Discount Percentage")

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

        # Make sure all features are numeric before predictions
        if all(features.select_dtypes(include=['object']).empty for features in [features_week, features_month, features_discount]):
            # Make predictions
            predictions_weekly = loaded_model_weekly.predict(features_week)
            predictions_monthly = loaded_model_monthly.predict(features_month)
            predictions_discount = loaded_model_discount.predict(features_discount)

            # Add predictions to the original data
            data['Next Week Sales'] = predictions_weekly.round()
            data['Predicted Next Month Sales'] = predictions_monthly.round()
            data['Predicted Discount (%)'] = predictions_discount.round()

            # Calculate the previous average sales for each product
            data['Current Avg Weekly Sales'] = data.groupby('Product Name')['Future_7_Day_Sales'].transform('mean').round()
            data['Current Avg Monthly Sales'] = data.groupby('Product Name')['Monthly Sales'].transform('mean').round()
            data['Current Avg Discount (%)'] = data.groupby('Product Name')['Discount (%)'].transform('mean').round()

            # Rename 'Stock After Sale' to 'Current Stock'
            data.rename(columns={'Stock After Sale': 'Current Stock'}, inplace=True)

            # Group the data by product and calculate the average predicted sales and previous average sales
            grouped_data = data.groupby('Product Name', as_index=False).agg({
                'Current Avg Weekly Sales': 'mean',
                'Next Week Sales': 'mean',
                'Current Avg Monthly Sales': 'mean',
                'Predicted Next Month Sales': 'mean',
                'Current Avg Discount (%)': 'mean',
                'Predicted Discount (%)': 'mean'
            }).round()

            # Display the comparison of previous average sales vs predicted sales
            st.success("Comparison of Current Average Weekly and Monthly Sales with Predicted Sales and Discount (%)")
            st.dataframe(grouped_data)

            # Stock Sufficiency Analysis using the latest entry for each product
            latest_data = data.sort_values('Date').drop_duplicates('Product Name', keep='last')

            # Calculate projected stock at the end of the next month
            latest_data['Projected Stock at End of Next Month'] = (latest_data['Current Stock'] - latest_data['Predicted Next Month Sales']).round()

            # Separate data into adequate and inadequate stock based on the projected stock
            adequate_stock = latest_data[latest_data['Projected Stock at End of Next Month'] >= 0]
            inadequate_stock = latest_data[latest_data['Projected Stock at End of Next Month'] < 0]

            # Display adequate and inadequate stock data
            st.header("Stock Sufficiency Analysis")
            st.success("Products with Adequate Stock for Next Month")
            st.dataframe(adequate_stock[['Product Name', 'Current Stock', 'Predicted Next Month Sales', 'Projected Stock at End of Next Month']])

            st.success("Products with Inadequate Stock for Next Month")
            st.dataframe(inadequate_stock[['Product Name', 'Current Stock', 'Predicted Next Month Sales', 'Projected Stock at End of Next Month']])

        else:
            st.error("Error: Non-numeric data found in features. Check the output for details.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
