import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="Advanced Analytics", layout="wide")

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

st.title("🔬 Advanced Analytics")
st.markdown("Apply statistical and machine learning techniques to uncover deeper patterns.")

# --- 1. Comparative Analysis (ANOVA) ---
st.header("1. Comparative Analysis (ANOVA)")
st.markdown("""
**ANOVA (Analysis of Variance)** tests whether there are any statistically significant differences 
between the means (averages) of two or more independent groups.
""")
st.info("For example: Is the *average deposit amount* significantly different between 'Rural', 'Semi-urban', and 'Urban' population groups?")

# User selection
anova_measure = st.selectbox("Select Measure to Compare:", numeric_cols, key="anova_m")
anova_group = st.selectbox("Select Grouping Column:", ['region', 'population_group'], key="anova_g")

if st.button("Run ANOVA Test"):
    # Clean formula for statsmodels (handles spaces and special chars)
    clean_measure = anova_measure.replace(" ", "_")
    clean_group = anova_group.replace(" ", "_")
    
    df_anova = df.rename(columns={anova_measure: clean_measure, anova_group: clean_group})

    # Fit the OLS (Ordinary Least Squares) model
    try:
        model = ols(f'Q("{clean_measure}") ~ C(Q("{clean_group}"))', data=df_anova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        st.subheader(f"ANOVA Results: {anova_measure} by {anova_group}")
        st.dataframe(anova_table)
        
        # Interpretation
        p_value = anova_table['PR(>F)'][0]
        st.markdown("**Interpretation:**")
        if p_value < 0.05:
            st.success(f"""
            The **p-value is {p_value:.4f} (which is < 0.05).**
            
            **Conclusion:** We **reject** the null hypothesis. There is a **statistically significant difference** in the average '{anova_measure}' between the different '{anova_group}' categories.
            """)
        else:
            st.warning(f"""
            The **p-value is {p_value:.4f} (which is >= 0.05).**
            
            **Conclusion:** We **fail to reject** the null hypothesis. There is **no statistically significant difference** in the average '{anova_measure}' between the different '{anova_group}' categories.
            """)
            
        # Show group means
        st.subheader("Group Averages")
        st.dataframe(df.groupby(anova_group)[anova_measure].mean().reset_index(name=f"Average {anova_measure}").sort_values(by=f"Average {anova_measure}", ascending=False))
        
    except Exception as e:
        st.error(f"An error occurred during ANOVA calculation: {e}")
        st.error("This can happen if a group has insufficient data. Try different inputs.")


# --- 2. Clustering (K-Means) & PCA ---
st.header("2. Customer/District Segmentation (K-Means Clustering)")
st.markdown("""
**Clustering** is an unsupervised machine learning technique that groups similar data points together.
Here, we can group **districts** or **states** based on their business metrics to identify distinct segments.
""")

# --- Setup for Clustering ---
cluster_level = st.selectbox("Select Level to Cluster:", ['district_name', 'state_name', 'region'])
st.markdown(f"We will aggregate all data to the **{cluster_level}** level, summing up all numeric metrics.")

# Aggregate data
df_cluster = df.groupby(cluster_level)[numeric_cols].sum().reset_index()

# Check for sufficient data
if len(df_cluster) < 2:
    st.warning(f"Not enough unique {cluster_level}s ({len(df_cluster)}) to perform clustering. Please select a different level.")
    st.stop()

st.info(f"Aggregated data to {len(df_cluster)} unique {cluster_level}s.")

# --- Feature Scaling ---
st.subheader("Step 1: Feature Scaling")
st.markdown("We must scale the features (e.g., 'no_of_accounts', 'deposit_amount') so that one metric doesn't dominate the others just because it has larger numbers.")
scaler = StandardScaler()
try:
    data_scaled = scaler.fit_transform(df_cluster[numeric_cols])
    st.success("Data successfully scaled using StandardScaler.")
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.info("This can happen with data that has zero variance (e.g., all values are the same).")
    st.stop()
    

# --- K-Means Clustering ---
st.subheader("Step 2: K-Means Clustering")
st.markdown("Select the number of clusters (K) you want to find.")
k = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=4)

# Run K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(data_scaled)
df_cluster['Cluster'] = df_cluster['Cluster'].astype(str) # For coloring

st.success(f"Successfully clustered the data into {k} segments.")

# --- PCA for Visualization ---
st.subheader("Step 3: Principal Component Analysis (PCA) for Visualization")
st.markdown("""
Since we can't plot 3+ dimensions (our numeric features) on a 2D chart, we use **PCA** to reduce these features into two 'Principal Components' (PCA1, PCA2) that capture the most variance. 
This lets us visualize the clusters.
""")

pca = PCA(n_components=2)
pca_components = pca.fit_transform(data_scaled)

df_cluster['PCA1'] = pca_components[:, 0]
df_cluster['PCA2'] = pca_components[:, 1]

st.write(f"""
**Variance Explained:**
- **PCA1** captures: **{pca.explained_variance_ratio_[0] * 100:.2f}%** of the variance.
- **PCA2** captures: **{pca.explained_variance_ratio_[1] * 100:.2f}%** of the variance.
- **Total:** The 2D plot below captures **{pca.explained_variance_ratio_.sum() * 100:.2f}%** of the total information from all numeric features.
""")

# --- Plot Clusters ---
st.subheader("Step 4: Cluster Visualization")
fig_pca = px.scatter(
    df_cluster,
    x='PCA1',
    y='PCA2',
    color='Cluster',
    hover_name=cluster_level,
    hover_data=numeric_cols,
    title=f"K-Means Clusters of {cluster_level.title()}s (Visualized with PCA)"
)
st.plotly_chart(fig_pca, use_container_width=True)

# --- Cluster Interpretation ---
st.subheader("Step 5: Cluster Interpretation")
st.markdown(f"These are the average values for each cluster. This helps us define a 'persona' for each segment.")
cluster_summary = df_cluster.groupby('Cluster')[numeric_cols].mean().reset_index()
st.dataframe(cluster_summary)

st.caption("""
**How to Read This:**
- Look for a cluster with high values across all metrics (e.g., "High-Value Hubs").
- Look for a cluster with low values (e.g., "Developing Areas").
- Look for a cluster with high accounts but low deposits (e.g., "Low Engagement").
- This segmentation can guide different business strategies for each group.
""")