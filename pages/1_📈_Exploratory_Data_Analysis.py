import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

# Caching function for data loading
@st.cache_data
def load_data():
    df = pd.read_csv("populationgroup-wise-deposits.csv")
    df = df.drop(columns=['id', 'state_code', 'district_code'], errors='ignore')
    numeric_cols = ['no_of_offices', 'no_of_accounts', 'deposit_amount']
    categorical_cols = ['year', 'state_name', 'district_name', 'region', 'population_group']
    df['year'] = df['year'].astype(str)
    df = df[~((df['no_of_offices'] == 0) & (df['no_of_accounts'] == 0) & (df['deposit_amount'] == 0))]
    return df, numeric_cols, categorical_cols

df, numeric_cols, categorical_cols = load_data()

if df is None:
    st.error("Data not loaded. Please check app.py.")
    st.stop()

st.title("📈 Exploratory Data Analysis (EDA)")
st.markdown("This section explores the characteristics of the dataset, including statistical summaries, distributions, and correlations.")

# --- 1. Descriptive Statistics ---
st.header("1. Descriptive Statistics")
st.markdown("Statistical summary for all key numeric business metrics.")
st.dataframe(df[numeric_cols].describe())

# --- 2. Categorical Value Counts ---
st.header("2. Categorical Data Frequency")
st.markdown("Shows the count of records for each category in the selected column.")
cat_col = st.selectbox("Select Categorical Column to Analyze:", categorical_cols)

if cat_col:
    st.subheader(f"Value Counts for: {cat_col.replace('_', ' ').title()}")
    counts_df = df[cat_col].value_counts().reset_index(name="Frequency")
    st.dataframe(counts_df)
    
    # Plot for low-cardinality columns
    if len(counts_df) < 50:
        fig = px.bar(counts_df.head(20), x=cat_col, y='Frequency', title=f"Top 20 Frequencies for {cat_col.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Skipping plot for {cat_col} due to high cardinality (over 50 unique values).")


# --- 3. Numeric Variable Distributions ---
st.header("3. Numeric Variable Distributions")
st.markdown("Visualize the distribution (Histogram), spread (Box Plot), and probability density (Violin Plot) of numeric variables.")
num_col = st.selectbox("Select Numeric Column to Visualize:", numeric_cols)

if num_col:
    st.subheader(f"Distribution Plots for: {num_col.replace('_', ' ').title()}")
    st.markdown(f"""
    - **Histogram**: Shows the frequency of data points falling into specified ranges (bins). It helps understand the central tendency, spread, and shape (e.g., skewness) of the data.
    - **Box Plot**: Displays the five-number summary (minimum, first quartile, median, third quartile, maximum). It's excellent for identifying the data's spread and potential outliers (points beyond the 'whiskers').
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(df, x=num_col, title=f"Histogram for {num_col}")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Insight: This histogram shows the frequency distribution. A 'right-skew' (long tail to the right) is common in financial data, indicating most values are small, but a few are very large.")

    with col2:
        # Box Plot
        fig_box = px.box(df, y=num_col, title=f"Box Plot for {num_col}")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Insight: The box represents the middle 50% of the data (IQR). The dots are outliers, which could be high-value districts/states or data anomalies.")

# --- 4. Correlation Analysis ---
st.header("4. Correlation Matrix")
st.markdown("""
This heatmap shows the Pearson correlation coefficient between numeric variables.
- Values near **+1** (dark red) indicate a strong **positive** correlation (as one variable increases, the other tends to increase).
- Values near **-1** (dark blue) indicate a strong **negative** correlation.
- Values near **0** (light color) indicate **no linear** correlation.
""")

corr = df[numeric_cols].corr()

# Plot using Seaborn
fig_corr, ax = plt.subplots()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap of Key Metrics")
st.pyplot(fig_corr)
st.caption("""
**Business Insight**: A strong positive correlation (e.g., > 0.8) between 'no_of_accounts' and 'deposit_amount' is expected. It confirms that as the number of accounts in an office/region grows, the total deposit amount also grows. A strong link between 'no_of_offices' and 'no_of_accounts' would also be logical.
""")

# --- 5. Pair Plots ---
st.header("5. Pair Plot Relationships")
st.markdown("""
A pair plot shows scatter plots for every pair of numeric variables and a distribution plot for each variable on the diagonal. 
It's a powerful tool for spotting relationships, clusters, and anomalies across multiple dimensions.
""")

# Optional: Add a grouping variable
pair_hue = st.selectbox("Select a grouping variable (hue) for the Pair Plot:", [None] + ['region', 'population_group'])

if st.button("Generate Pair Plot"):
    st.info(f"Generating pair plot... This might take a moment, especially with a hue variable.")
    # Create a smaller sample for faster plotting if 'hue' is used
    plot_df = df
    if pair_hue:
        # Sample to avoid overplotting and slow render
        plot_df = df.sample(n=min(5000, len(df)), random_state=1)
    
    fig_pair = sns.pairplot(plot_df, vars=numeric_cols, hue=pair_hue, corner=True)
    fig_pair.fig.suptitle("Pair Plot of Key Numeric Metrics", y=1.02)
    st.pyplot(fig_pair)
    st.caption(f"""
    **Business Insight**: Look for patterns. 
    - Do 'deposit_amount' and 'no_of_accounts' show a clear linear relationship? 
    - If grouped by '{pair_hue}', do different groups (e.g., 'Rural' vs 'Urban') show different relationships? For instance, 'Urban' areas (blue dots, if selected) might show a steeper slope, indicating more deposits per account.
    """)