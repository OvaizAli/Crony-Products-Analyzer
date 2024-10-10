import streamlit as st
import pandas as pd

# Title of the application
st.title("Crony Product Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)
    st.error("Data Preview")
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
