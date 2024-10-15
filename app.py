import streamlit as st
import pandas as pd

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

    # Ensure 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Day of Week Analysis
    st.success("Total Sales by Day of the Week")
    data['Day of Week'] = data['Date'].dt.day_name()  # Extract day of the week
    sales_by_day_of_week = data.groupby('Day of Week')['Total Sales ($)'].sum().reset_index().sort_values(by='Total Sales ($)', ascending=False).reset_index()
    st.dataframe(sales_by_day_of_week[['Day of Week', 'Total Sales ($)']])

    # Weekly Analysis
    st.success("Total Sales by Week")
    data['Week'] = data['Date'].dt.isocalendar().week  # Extract week number
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
    data['Profit Margin'] = data['Net Sales Value ($)'] / data['Total Sales ($)']  # Corrected profit margin formula
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

    # Prepare data for correlation analysis
    data['Day of Week'] = data['Date'].dt.dayofweek  # Monday = 0, Sunday = 6
    data['Week'] = data['Date'].dt.isocalendar().week  # Week number
    data['Is Weekend'] = data['Day of Week'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if weekend, 0 if weekday

    # Encode categorical variables (like Weather Condition and Holiday)
    data['Weather Condition Encoded'] = pd.factorize(data['Weather Condition'])[0]
    data['Is Holiday Encoded'] = data['Is Holiday'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Select relevant features and target variable
    features = ['Day of Week', 'Week', 'Is Weekend', 'Weather Condition Encoded', 'Is Holiday Encoded', 'Discount (%)', 'Total Sales ($)']
    correlation_matrix = data[features].corr()  # Calculate the correlation matrix

    # Extract the correlation of each feature with 'Total Sales ($)'
    sales_correlation = correlation_matrix[['Total Sales ($)']].sort_values(by='Total Sales ($)', ascending=False)

    # Combine the explanation in one text block
    explanation = ""

    for feature in sales_correlation.index:
        if feature != 'Total Sales ($)':  # Exclude Total Sales itself
            correlation_value = sales_correlation.loc[feature, 'Total Sales ($)']
            if correlation_value > 0:
                explanation += (f"- **{feature}** has a **positive** correlation of {correlation_value:.2f} with total sales. "
                                f"This means that as {feature} increases, total sales tend to increase. Leverage this factor to boost sales.\n")
            elif correlation_value < 0:
                explanation += (f"- **{feature}** has a **negative** correlation of {correlation_value:.2f} with total sales. "
                                f"As {feature} increases, total sales tend to decrease. Consider strategies to mitigate this factor's negative impact.\n")
            else:
                explanation += (f"- **{feature}** has a **neutral** correlation with total sales, meaning it has little to no effect on sales.\n")

    # Display the combined explanation
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
