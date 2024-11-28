import streamlit as st
import pandas as pd
import joblib
import os
import openai
import json
# from dotenv import load_dotenv
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


# load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN") 
api_key = st.secrets["general"]["OPENAI_API_TOKEN"]
openai.api_key = api_key

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
    st.success("Total Sales by Week of Month")
    # Add columns for year, month, and week_of_month
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week

    # Add column for Week_of_Month
    data['Week_of_Month'] = ((data['Date'].dt.day - 1) // 7) + 1

    # Only keep weeks 1 to 4 (for each month)
    data = data[data['Week_of_Month'] <= 4]

    # Group by Week_of_Month to get total sales for each week of the month
    sales_by_week_of_month = data.groupby('Week_of_Month')['Total Sales ($)'].sum().reset_index()

    # Sort the sales by total sales in descending order
    sales_by_week_of_month = sales_by_week_of_month.sort_values(by='Total Sales ($)', ascending=False)

    # Display the dataframe with total sales by week of the month
    st.dataframe(sales_by_week_of_month)
    
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

    st.success("Weather Condition and Temperature Analysis on Sales")

    # Define temperature bins for analysis
    data['Temperature Bin'] = pd.cut(data['Temperature (°C)'], bins=[-10, 10, 20, 30, 40], labels=['Cold', 'Mild', 'Warm', 'Hot'])

    # Group sales by Weather Condition and Temperature Bin
    weather_temp_sales = data.groupby(['Weather Condition', 'Temperature Bin'])['Total Sales ($)'].sum().unstack(fill_value=0)

    # Display sales distribution across weather conditions and temperature ranges
    st.dataframe(weather_temp_sales)

    
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

    st.success("Seasonal Demand Forecasting by Category")

    # Aggregate data by Category and Season, calculating total sales for each
    seasonal_sales = data.groupby(['Category', 'Season'])['Total Sales ($)'].sum().unstack(fill_value=0)

    # Calculate percentage change season-over-season for each category
    seasonal_sales_change = seasonal_sales.pct_change(axis=1) * 100
    seasonal_sales_change = seasonal_sales_change.round(2)

    # Display seasonal sales trends and growth percentage
    st.dataframe(seasonal_sales)
    st.success("Percentage change season-over-season for each category")
    st.dataframe(seasonal_sales_change)

    st.success("Product Return Rate Analysis (Top 5)")

    # Calculate the return rate as a percentage of items sold
    data['Return Rate (%)'] = (data['Returned'] / data['Quantity Sold']) * 100
    return_analysis = data.groupby('Product Name')['Return Rate (%)'].mean().reset_index()
    return_analysis = return_analysis.sort_values(by='Return Rate (%)', ascending=False).head(5)

    # Display top products with highest return rates
    st.dataframe(return_analysis)


    # Stock Management and Inventory Turnover Analysis
    st.success("Top 5 Products by their Current Stock")

    # Rename 'Stock After Sale' to 'Current Stock'
    data.rename(columns={'Stock After Sale': 'Current Stock'}, inplace=True)
    stock_data = data.groupby('Product Name')['Current Stock'].mean().reset_index()
    top_stock = stock_data.sort_values(by='Current Stock', ascending=False).head(5)
    st.dataframe(top_stock)
    
    st.success("Monthly Inventory Turnover Rate Analysis")

    # Calculate monthly average stock and turnover
    monthly_stock_turnover = data.groupby('Month').apply(
        lambda x: (x['Total Sales ($)'].sum() / x['Current Stock'].mean()).round(2)
    ).reset_index(name='Inventory Turnover Rate')

    # Display monthly turnover rates
    st.dataframe(monthly_stock_turnover)


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

    # Generate association rules with lift metric, including num_itemsets
    num_itemsets = frequent_items.shape[0]

    # Generate association rules with lift metric
    rules = association_rules(frequent_items, metric="lift", min_threshold=1, num_itemsets=num_itemsets)

    # Display only rules with two or more items in the antecedents or consequents
    filtered_rules = rules[(rules['antecedents'].apply(len) > 1) | (rules['consequents'].apply(len) > 1)]

    # Display results in Streamlit
    st.success("Products Often Bought Together")

    # Check if there are no rules and display an appropriate message
    if filtered_rules.empty:
        st.warning("We couldn't find any products that are frequently bought together.")
    else:
        # Convert frozenset to a string of product names for better readability
        filtered_rules['Items Purchased Together'] = filtered_rules['antecedents'].apply(lambda x: ', '.join(x))
        filtered_rules['Suggested Products'] = filtered_rules['consequents'].apply(lambda x: ', '.join(x))
    
        # Ensure we only select the relevant columns for renaming
        filtered_rules = filtered_rules[['Items Purchased Together', 'Suggested Products', 'support', 'lift', 'confidence']]
        
        # Rename columns for better understanding
        filtered_rules.columns = [
            'Items Purchased Together',      # Represents the products that were bought together
            'Suggested Products',             # Represents products recommended based on past purchases
            'Purchase Frequency',             # Support indicates how frequently this product combination was purchased
            'Association Strength',           # Indicates the strength of the association between the products based on purchasing patterns
            'Likelihood (%)'                  # The likelihood of purchasing the suggested products given that the items purchased together are bought
        ]

        # Calculate 'Purchase Frequency (%)' if not directly available
        total_transactions = data['Transaction ID'].nunique()
        filtered_rules['Purchase Frequency (%)'] = (filtered_rules['Purchase Frequency'] * 100).round(2)

        # Calculate 'Likelihood (%)' based on confidence
        filtered_rules['Likelihood (%)'] = (filtered_rules['Likelihood (%)'] * 100).round(2)

        # Sort the filtered rules by Association Strength and get the top N
        top_n = 10  # You can change this number to get more or fewer results
        top_filtered_rules = filtered_rules.nlargest(top_n, 'Association Strength')  # Sort by 'Strength of Association' or another column

        # Display the top filtered rules
        st.dataframe(top_filtered_rules[['Items Purchased Together', 'Suggested Products', 'Purchase Frequency (%)', 'Likelihood (%)', 'Association Strength']].sort_values(by='Association Strength', ascending=False))

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

    st.success("Promotion Strategy Analysis Using Price Elasticity and Profitability")

    # Filter products with high price sensitivity (elasticity) and high profitability
    high_elasticity_profitable = price_sensitivity_analysis[
        (price_sensitivity_analysis['Price Sensitivity Ratio'] > 1) &
        (price_sensitivity_analysis['Price Change (%)'] < 0)
    ]

    # Check the results before merging
    if high_elasticity_profitable.empty:
        st.warning("No products meet the high price sensitivity and discount criteria.")
    else:
        st.write("Filtered products before merge:", high_elasticity_profitable)

        # Proceed with the merge operation if filtered results are not empty
        high_elasticity_profitable = high_elasticity_profitable.merge(
            data[['Product Name', 'Profit Margin']], on='Product Name', how='inner'
        ).drop_duplicates()
        
        if high_elasticity_profitable.empty:
            st.warning("No products found after merging with profit margin data.")
        else:
            # Sort by Profit Margin and display top 10
            high_elasticity_profitable = high_elasticity_profitable.sort_values(by='Profit Margin', ascending=False).head(10)
            st.dataframe(high_elasticity_profitable)

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

    # Identify the categorical columns in your DataFrame
    categorical_cols = new_data.select_dtypes(include='category').columns

    # Add `0` as a category to all categorical columns
    for col in categorical_cols:
        new_data[col] = new_data[col].cat.add_categories([0])
        
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


            # Group the data by product and calculate the average predicted sales and previous average sales
            grouped_data = data.groupby('Product Name', as_index=False).agg({
                'Current Avg Weekly Sales': 'mean',
                'Next Week Sales': 'mean',
                # 'Current Avg Monthly Sales': 'mean',
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


    data['Week'] = data['Date'].dt.isocalendar().week  # Adjust if using 'Month' for monthly clustering
    weekly_sales = data.groupby(['Product Name', 'Week'])['Total Sales ($)'].sum().unstack(fill_value=0)

    # Step 2: Scale the data for clustering
    scaler = StandardScaler()
    scaled_sales = scaler.fit_transform(weekly_sales.T).T  # Transpose for correct scaling per product

    # Step 3: Perform KMeans clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    weekly_sales['Cluster'] = kmeans.fit_predict(scaled_sales)

    # Define descriptive names for each cluster
    cluster_names = {
        0: "High Demand Products",
        1: "Seasonal Products",
        2: "Low Demand Products"
    }

    # Map cluster numbers to names in weekly_sales DataFrame
    weekly_sales['Cluster Name'] = weekly_sales['Cluster'].map(cluster_names)

    # Step 4: Add cluster labels back to the main data
    data = data.merge(weekly_sales[['Cluster']], on='Product Name', how='left')

    # Step 5: Organize products by clusters with descriptive names
    clustered_products = {
        cluster_names[i]: list(weekly_sales[weekly_sales['Cluster'] == i].index)
        for i in range(n_clusters)
    }
    clustered_products_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in clustered_products.items()]))

    # Display the DataFrame with products grouped by descriptive cluster names
    st.success("Product Segmentation Based on Sales Patterns")
    st.dataframe(clustered_products_df)

    results = {
    "sales_by_day_of_week": data.groupby('Day of Week')['Total Sales ($)'].sum(),
    "profit_analysis": data.groupby('Product Name')['Profit Margin'].mean(),
    "weather_sales": data.groupby('Weather Condition')['Total Sales ($)'].sum(),
    "correlation_matrix": correlation_matrix,
    "seasonal_sales": seasonal_sales,
    "return_analysis": return_analysis,
    "monthly_stock_turnover": monthly_stock_turnover,
    "association_rules": rules,
    "predicted_data": data[['Product Name', 'Next Week Sales', 'Predicted Next Month Sales', 'Predicted Discount (%)']],
    "clustered_data": weekly_sales
    }

    # Function to generate analysis report using LLM
    def generate_sales_report(results):
        # Construct the enhanced prompt using the results data
        prompt = f"""
        You are an expert retail sales analyst. Based on the provided data, generate a **comprehensive performance analysis report** that includes detailed insights, specific details or calculations, observations, and actionable recommendations. Here's the data:

        1. **Sales by Day of Week**:
        Analyze how sales vary by day of the week. Identify peak and low sales days, and assess the performance of weekdays versus weekends.
        {results['sales_by_day_of_week'].to_string()}

        2. **Profit Analysis by Product**:
        Evaluate the profitability of products. Highlight products with high-profit margins and those with low margins requiring optimization.
        {results['profit_analysis'].to_string()}

        3. **Total Sales by Weather Condition**:
        Examine the impact of weather conditions on sales performance. Identify which weather conditions drive higher sales and recommend strategies for weather-driven promotions.
        {results['weather_sales'].to_string()}

        4. **Correlation Analysis**:
        Interpret the correlation between key factors (e.g., discounts, sales volume, weather, holidays). Highlight significant positive or negative relationships and explain their implications.
        {results['correlation_matrix']}

        5. **Seasonal Sales Performance**:
        Analyze the performance of different product categories across seasons. Identify seasonal trends, and suggest strategies to capitalize on peak seasons or improve off-season sales.
        {results['seasonal_sales']}

        6. **Top 5 Products by Return Rate**:
        Investigate products with the highest return rates. Highlight potential reasons (e.g., quality issues, mismatch with customer expectations) and recommend corrective actions.
        {results['return_analysis']}

        7. **Monthly Sales Turnover**:
        Assess monthly sales turnover ratios. Identify months with strong or weak performance and propose inventory strategies to optimize stock management.
        {results['monthly_stock_turnover']}

        8. **Association Rules (Product Pairing Insights)**:
        Identify frequent product pairings and their lift metrics. Provide insights on how these pairings can be leveraged for cross-selling or bundling strategies.
        {results['association_rules']}

        9. **Predicted Next Week Sales & Discounts**:
        Analyze predicted sales for the next week and next month along with suggested discount percentages. Assess how these predictions align with past trends and suggest strategies to maximize profitability.
        {results['predicted_data'].to_string()}

        10. **Sales Clusters (Product Week-wise Sales)**:
        Analyze clusters of products based on weekly sales patterns. Identify characteristics of high-performing clusters and underperforming ones, and recommend tailored strategies for each.

        Based on the above data, provide the following in your analysis:
        1. **Store Performance Rating**: Rate overall sales performance as excellent, good, average, or poor, with a justification for your rating.
        2. **Key Insights**: Highlight the top 5 most notable insights or observations from the analysis.
        3. **Actionable Recommendations**: Provide 5 detailed, specific recommendations to improve or sustain sales performance.
        4. **Challenges**: Identify any challenges or bottlenecks observed in the data and suggest ways to address them.

        Format your response in structured JSON as follows:
        {{
            "store_performance_rating": "string",
            "key_insights": ["string", "string", "string", "string", "string"],
            "recommendations": ["string", "string", "string", "string", "string"],
            "challenges": ["string", "string"]
        }}
        """

        try:
            # Generate a completion using the LLM (replace with actual model)
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Replace with the correct model
                messages=[{"role": "system", "content": "You are a helpful assistant for analyzing retail sales performance."},
                        {"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7
            )

            # Extract response text
            response_text = response.choices[0].message['content'].strip()

            # Clean up the response text to handle potential formatting issues
            cleaned_response = response_text.strip().strip('```json').strip('```')

            # Attempt to parse the cleaned response as JSON
            try:
                # Parse the cleaned response as JSON
                report_data = json.loads(cleaned_response)
                
                # Return the raw JSON (no DataFrame conversion)
                return report_data
            except json.JSONDecodeError as e:
                print(f"Error: Response is not valid JSON - {cleaned_response}, Error: {e}")
                # Return a default error message in case of invalid JSON
                return {
                    "store_performance_rating": "unknown",
                    "key_insights": ["Error occurred while processing data"],
                    "recommendations": [],
                    "challenges": ["Unable to analyze due to processing error."]
                }

        except Exception as e:
            print(f"Error processing results: {e}")
            # Return a default error message in case of failure
            return {
                "store_performance_rating": "unknown",
                "key_insights": ["Error occurred while processing data"],
                "recommendations": [],
                "challenges": ["Unable to analyze due to processing error."]
            }

    # Generate the report and display it
    report = generate_sales_report(results)

    # Print or display the report in Streamlit
    st.title("Sales Performance Analysis Report")
    if report:
        st.subheader(f"Store Performance Rating - {report['store_performance_rating'].capitalize()}")
        st.subheader("Key Insights")
        for insight in report['key_insights']:
            st.write(f"- {insight}")
        st.subheader("Actionable Recommendations")
        for recommendation in report['recommendations']:
            st.write(f"- {recommendation}")
        st.subheader("Challenges")
        for challenge in report['challenges']:
            st.write(f"- {challenge}")
    else:
        st.error("Failed to generate report.")