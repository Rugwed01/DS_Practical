import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Segmentation & Aggregations", layout="wide")

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

st.title("📊 Segmentation & Business Aggregations")
st.markdown("Analyze performance by filtering and grouping data across different business dimensions.")

# --- Sidebar Filters ---
st.sidebar.header("Master Filters")
st.sidebar.markdown("Use these filters to control the data displayed in all charts on this page.")

# Get unique, sorted values
years = sorted(df['year'].unique())
regions = sorted(df['region'].unique())
pop_groups = sorted(df['population_group'].unique())
states = sorted(df['state_name'].unique())

# Create filters
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)
selected_pop_groups = st.sidebar.multiselect("Select Population Group(s)", pop_groups, default=pop_groups)
selected_states = st.sidebar.multiselect("Select State(s)", states, default=states)

# Filter the dataframe based on selections
filtered_df = df[
    df['year'].isin(selected_years) &
    df['region'].isin(selected_regions) &
    df['population_group'].isin(selected_pop_groups) &
    df['state_name'].isin(selected_states)
]

if filtered_df.empty:
    st.warning("No data matches your filter criteria. Please adjust the filters in the sidebar.")
    st.stop()

# --- 1. KPIs ---
st.header("1. Key Performance Indicators (Filtered)")
st.markdown("These KPIs reflect the totals for your current filter selections.")

kpi_deposits = filtered_df['deposit_amount'].sum()
kpi_accounts = filtered_df['no_of_accounts'].sum()
kpi_offices = filtered_df['no_of_offices'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Deposit Amount", f"{kpi_deposits:,.0f}")
col2.metric("Total Number of Accounts", f"{kpi_accounts:,.0f}")
col3.metric("Total Number of Offices", f"{kpi_offices:,.0f}")

# --- 2. Trend Analysis ---
st.header("2. Trend Analysis")
st.markdown("Shows the trend of a selected metric over time, aggregated by year.")

trend_measure = st.selectbox("Select Measure for Trend Analysis:", numeric_cols)

if trend_measure:
    # Group by year and sum the selected measure
    df_trend = filtered_df.groupby('year')[trend_measure].sum().reset_index()
    
    fig_trend = px.line(
        df_trend,
        x='year',
        y=trend_measure,
        title=f"Trend of {trend_measure.replace('_', ' ').title()} Over Time",
        markers=True
    )
    fig_trend.update_layout(xaxis_title="Year", yaxis_title=f"Total {trend_measure.replace('_', ' ').title()}")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Insight: This chart helps identify growth or decline year-over-year for the selected metric and filters. Is the growth accelerating or slowing down?")

# --- 3. Segmentation & Share Analysis ---
st.header("3. Segmentation & Share Analysis")
st.markdown("Breaks down key metrics by a chosen business dimension (e.g., Region, Population Group).")

seg_dimension = st.selectbox("Select Dimension to Group By:", ['region', 'population_group', 'state_name'])
seg_measure = st.selectbox("Select Measure to Analyze:", numeric_cols)

if seg_dimension and seg_measure:
    # Group by the selected dimension
    df_segment = filtered_df.groupby(seg_dimension)[seg_measure].sum().reset_index().sort_values(by=seg_measure, ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar Chart
        st.subheader(f"Total {seg_measure.replace('_', ' ').title()} by {seg_dimension.replace('_', ' ').title()}")
        fig_bar = px.bar(
            df_segment.head(20),  # Show top 20
            x=seg_dimension,
            y=seg_measure,
            title=f"Top 20 {seg_dimension.replace('_', ' ').title()}s by {seg_measure.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Insight: This bar chart identifies the top-performing segments. Where is the bulk of the business concentrated?")

    with col2:
        # Pie Chart
        st.subheader(f"Share of {seg_measure.replace('_', ' ').title()} by {seg_dimension.replace('_', ' ').title()}")
        fig_pie = px.pie(
            df_segment.head(10),  # Show top 10 for pie clarity
            names=seg_dimension,
            values=seg_measure,
            title=f"Market Share of {seg_measure.replace('_', ' ').title()} (Top 10)"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Insight: This pie chart shows the relative contribution of each segment. Are you overly dependent on one or two segments?")

# --- 4. Top N / Bottom N Analysis ---
st.header("4. Top & Bottom Performers")
st.markdown("Identify the best and worst performing districts or states based on your chosen metric.")

top_n_dim = st.selectbox("Select Dimension to Rank:", ['district_name', 'state_name'])
top_n_measure = st.selectbox("Select Measure to Rank by:", numeric_cols, key='topn_m')
n_val = st.slider("Select 'N' (Number of performers to show):", min_value=3, max_value=30, value=10)

if top_n_dim and top_n_measure:
    # Group by the dimension
    df_grouped_n = filtered_df.groupby(top_n_dim)[top_n_measure].sum().reset_index()
    
    # Get Top N and Bottom N
    df_top_n = df_grouped_n.nlargest(n_val, top_n_measure)
    df_bottom_n = df_grouped_n.nsmallest(n_val, top_n_measure).sort_values(by=top_n_measure, ascending=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Top {n_val} {top_n_dim.replace('_', ' ').title()}s")
        fig_top = px.bar(df_top_n, x=top_n_dim, y=top_n_measure, title=f"Top {n_val} Performers")
        fig_top.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_top, use_container_width=True)
        st.dataframe(df_top_n)

    with col2:
        st.subheader(f"Bottom {n_val} {top_n_dim.replace('_', ' ').title()}s")
        fig_bottom = px.bar(df_bottom_n, x=top_n_dim, y=top_n_measure, title=f"Bottom {n_val} Performers")
        fig_bottom.update_layout(xaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bottom, use_container_width=True)
        st.dataframe(df_bottom_n)

    st.caption("Insight: Top performers represent key markets. Bottom performers may represent untapped opportunities or areas needing strategic intervention.")

# --- 5. Dynamic Data Table ---
st.header("5. Filtered Data Explorer")
st.markdown("View the raw data table based on your current filter selections.")
st.dataframe(filtered_df)

# Download Button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_bank_data.csv",
    mime="text/csv",
)