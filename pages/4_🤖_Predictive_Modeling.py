import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set page config
st.set_page_config(page_title="Predictive Modeling", layout="wide")

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

st.title("🤖 Predictive Modeling")
st.markdown("Using machine learning to predict key metrics and classify business performance.")
st.info("Note: These are simplified models for demonstration. Real-world modeling requires rigorous feature engineering, hyperparameter tuning, and cross-validation.")

# --- Part 1: Regression (Predicting a Number) ---
st.header("Part 1: Regression Models")
st.markdown("Predicting a continuous value, like `deposit_amount` or `no_of_accounts`.")

# --- Model Setup ---
st.subheader("Model Setup")
target_var = st.selectbox(
    "Select Target Variable (What to predict):",
    ['deposit_amount', 'no_of_accounts']
)

# Define features
# Use a subset of features for simplicity
numeric_features = ['no_of_offices']
categorical_features = ['region', 'population_group', 'year']

# Remove target from features if it's selected as one (e.g., predicting accounts using offices)
if target_var in numeric_features:
    numeric_features.remove(target_var)

all_features = numeric_features + categorical_features
st.write(f"**Target:** `{target_var}`")
st.write(f"**Features:** `{', '.join(all_features)}`")

# Prepare data
X = df[all_features]
y = df[target_var]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pre-processing pipeline
# 1. OneHotEncoder for categorical features
# 2. StandardScaler for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
}

results = []

# --- Train and Evaluate Models ---
st.subheader("Model Performance Comparison")
with st.spinner("Training and evaluating all regression models..."):
    for name, model in models.items():
        # Create full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            "Model": name,
            "R-squared ($R^2$)": r2,
            "Mean Absolute Error (MAE)": mae,
            "Root Mean Squared Error (RMSE)": rmse,
            "pipeline": pipeline, # Save pipeline for later
            "predictions": y_pred
        })

    results_df = pd.DataFrame(results).drop(columns=['pipeline', 'predictions'])
    st.dataframe(results_df.sort_values(by="R-squared ($R^2$)", ascending=False))

    st.markdown("""
    **Metric Explanations:**
    - **R-squared ($R^2$)**: The proportion of the variance in the target variable that is predictable from the features. (1 is a perfect prediction, 0 means the model is no better than just guessing the average). **Higher is better.**
    - **MAE**: The average absolute difference between the predictions and the actual values. **Lower is better.**
    - **RMSE**: The square root of the average of squared differences. It penalizes large errors more. **Lower is better.**
    """)

# --- Get Best Model ---
best_model_result = max(results, key=lambda x: x['R-squared ($R^2$)'])
st.success(f"**Best Model:** `{best_model_result['Model']}` (based on $R^2$)")

best_pipeline = best_model_result['pipeline']
best_predictions = best_model_result['predictions']

# --- Diagnostic Plots ---
st.subheader(f"Diagnostic Plots for Best Model ({best_model_result['Model']})")

col1, col2 = st.columns(2)

with col1:
    # 1. Residual Plot
    residuals = y_test - best_predictions
    fig_res = px.scatter(
        x=best_predictions,
        y=residuals,
        title="Residual Plot (Predicted vs. Residuals)",
        labels={'x': 'Predicted Value', 'y': 'Residuals'},
        opacity=0.5
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_res, use_container_width=True)
    st.caption("Insight: Ideally, residuals should be randomly scattered around the red line (zero) with no clear pattern. Patterns (like a funnel shape) suggest the model may be violating some assumptions (heteroscedasticity).")

with col2:
    # 2. Prediction Error Plot
    fig_pred = px.scatter(
        x=y_test,
        y=best_predictions,
        title="Prediction Error Plot (Actual vs. Predicted)",
        labels={'x': 'Actual Value', 'y': 'Predicted Value'},
        opacity=0.5
    )
    # Add 45-degree line
    fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                       x1=y_test.max(), y1=y_test.max(),
                       line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_pred, use_container_width=True)
    st.caption("Insight: A perfect model would have all dots on the red 45-degree line. The closer the dots are to this line, the more accurate the predictions.")

# --- Feature Importance (for Random Forest) ---
if best_model_result['Model'] == 'Random Forest Regressor':
    st.subheader("Feature Importances (from Random Forest)")
    st.markdown("Shows which features the model found most predictive.")
    
    # Get feature names from preprocessor
    try:
        model_rf = best_pipeline.named_steps['model']
        preprocessor_rf = best_pipeline.named_steps['preprocessor']
        
        # Get one-hot encoded feature names
        cat_features_out = preprocessor_rf.named_transformers_['cat'].get_feature_names_out(categorical_features)
        
        # Combine all feature names
        all_feature_names = numeric_features + list(cat_features_out)
        
        importances = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': model_rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Plot
        fig_imp = px.bar(importances.head(20), x='Feature', y='Importance', title="Top 20 Most Important Features")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Insight: High-importance features (e.g., `no_of_offices`, `population_group_Urban`) are the strongest drivers of the prediction.")

    except Exception as e:
        st.warning(f"Could not extract feature importances: {e}")


# --- Part 2: Classification (Predicting a Category) ---
st.header("Part 2: Classification Models")
st.markdown(f"Classifying if a record has 'High' or 'Low' **{target_var}** based on its median value.")

# --- Setup Classification ---
st.subheader("Classification Model Setup")
median_val = y.median()
st.info(f"Defining 'High' as any value > median ({median_val:,.2f}) and 'Low' as any value <= median.")

# Create binary target
y_class = (y > median_val).astype(int) # 1 for High, 0 for Low

# Train-test split for classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.3, random_state=42, stratify=y_class)

# --- Train Logistic Regression ---
st.subheader("Logistic Regression Performance")
with st.spinner("Training Logistic Regression model..."):
    # Use the same pipeline, just swap the model
    log_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('model', LogisticRegression(max_iter=1000, random_state=42))])
    
    log_reg_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = log_reg_pipeline.predict(X_test_c)
    
    # --- Evaluate ---
    st.write(f"**Accuracy:** {accuracy_score(y_test_c, y_pred_c):.3f}")
    st.markdown("*(Accuracy is the % of correct predictions)*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Classification Report
        st.text("Classification Report:")
        report = classification_report(y_test_c, y_pred_c, target_names=['Low', 'High'])
        st.text(report)
        st.caption("Precision: Of all 'High' predictions, what % was correct? \nRecall: Of all actual 'High' values, what % did the model find?")

    with col2:
        # Confusion Matrix
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test_c, y_pred_c)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Low', 'High'],
            y=['Low', 'High'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(f"Top-Left (True Negative): Correctly predicted 'Low'. \nBottom-Right (True Positive): Correctly predicted 'High'.")