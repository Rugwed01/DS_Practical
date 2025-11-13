import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Bank Deposit Analysis Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Caching function for data loading
@st.cache_data
def load_data():
    """
    Loads, cleans, and prepares the dataset.
    This function is cached to ensure data is loaded only once.
    """
    try:
        df = pd.read_csv("populationgroup-wise-deposits.csv")
    except FileNotFoundError:
        st.error(f"Error: The file 'populationgroup-wise-deposits.csv' was not found.")
        st.info("Please make sure the data file is in the same directory as app.py.")
        return None, None, None

    # Drop non-essential ID/code columns as names are present
    df = df.drop(columns=['id', 'state_code', 'district_code'], errors='ignore')
    
    # Define column types
    numeric_cols = ['no_of_offices', 'no_of_accounts', 'deposit_amount']
    # 'year' is treated as categorical for grouping/filtering
    categorical_cols = ['year', 'state_name', 'district_name', 'region', 'population_group']
    
    # Convert year to string to treat it as a category
    df['year'] = df['year'].astype(str)
    
    # Filter out rows where all key numeric values are zero (potential data entry issues)
    df = df[~((df['no_of_offices'] == 0) & (df['no_of_accounts'] == 0) & (df['deposit_amount'] == 0))]
    
    return df, numeric_cols, categorical_cols

# Load the data
df, numeric_cols, categorical_cols = load_data()

# Main page content
if df is not None:
    st.title("🏦 Bank Deposit Analysis Dashboard")
    st.markdown("""
    Welcome to the Bank Deposit Analysis Dashboard. This tool provides a comprehensive, 
    business-oriented analysis of bank deposits, accounts, and offices across various 
    regions, states, districts, and population groups in India.

    Use the navigation menu on the left to explore different sections:
    - **Home**: This page, showing an overview and data summary.
    - **Exploratory Data Analysis (EDA)**: Deep dive into variable distributions, correlations, and outliers.
    - **Segmentation & Aggregations**: Analyze business performance by key segments (time, geography, etc.).
    - **Advanced Analytics**: Uncover deeper insights using clustering, PCA, and statistical tests.
    - **Predictive Modeling**: Forecast key metrics and understand their drivers using machine learning.
    """)

    st.header("High-Level Business KPIs")
    st.markdown("These metrics represent the grand total across all years and regions in the dataset.")
    
    # Calculate total KPIs
    total_deposits = df['deposit_amount'].sum()
    total_accounts = df['no_of_accounts'].sum()
    total_offices = df['no_of_offices'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Deposit Amount (in units)", f"{total_deposits:,.0f}")
    col2.metric("Total Number of Accounts", f"{total_accounts:,.0f}")
    col3.metric("Total Number of Offices", f"{total_offices:,.0f}")

    st.header("Data Preview")
    st.markdown("Here is a random sample of 10 rows from the dataset:")
    st.dataframe(df.sample(10))

    st.header("Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Data Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe())
    
    with col2:
        st.subheader("Categorical Data Summary")
        for col in ['region', 'population_group', 'year']:
            st.markdown(f"**{col.replace('_', ' ').title()}**")
            st.dataframe(df[col].value_counts().reset_index(name="Count"))

    st.header("Geographic Distribution of Deposits")
    st.markdown("Sum of deposits by state. Darker colors indicate higher deposit amounts.")
    
    # Group by state for a map-like bar chart
    state_deposits = df.groupby('state_name')['deposit_amount'].sum().reset_index().sort_values(by='deposit_amount', ascending=False)
    
    fig = px.bar(
        state_deposits.head(20),  # Show top 20 for clarity
        x='state_name',
        y='deposit_amount',
        title="Top 20 States by Total Deposit Amount",
        labels={'state_name': 'State', 'deposit_amount': 'Total Deposit Amount'}
    )
    fig.update_layout(xaxis_title="State", yaxis_title="Total Deposit Amount")
    st.plotly_chart(fig, use_container_width=True)
    
    st.sidebar.success("Select an analysis page above.")

else:
    st.stop()