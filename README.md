# 🏦 Bank Deposit Analysis Dashboard

A Streamlit-powered interactive dashboard for analyzing **bank deposits, accounts, and offices** across India, segmented by **state, district, region, year, and population group**.

This project transforms raw deposit data into meaningful **business intelligence** through KPIs, EDA, segmentation insights, and visual analytics.

---

## Streamlit Deployed Link
<https://dspractical-mini-project.streamlit.app>

---

## 📌 **Project Overview**

This dashboard provides a comprehensive analytical interface to explore India’s banking deposit landscape.
Using the dataset *populationgroup-wise-deposits.csv*, the application allows users to:

* Visualize deposit distribution across states and population groups
* Explore key business metrics such as total deposits, accounts, and offices
* Perform quick data diagnosis through descriptive statistics and previews
* Understand geographic patterns through interactive charts
* Navigate multiple analysis sections (EDA, segmentation, ML models — modular support ready)

The app is built entirely using **Streamlit**, **Pandas**, and **Plotly**.

---

## 🗂️ **Dataset Description**

**File:** `cleaned-populationgroup-wise-deposits.csv`
Contains bank-related metrics across India grouped by:

* **Year**
* **State Name**
* **District Name**
* **Region**
* **Population Group** (Rural, Urban, Metro, etc.)
* **Numeric Features:**

  * Number of Offices
  * Number of Accounts
  * Deposit Amount

During preprocessing (inside `load_data()`):

* Non-essential ID columns are dropped
* Year is converted to categorical
* Rows with all-zero numerical values are removed
* Data is cached for faster re-runs

---

## 🚀 **Features**

### ✅ **1. High-Level KPIs**

* Total Deposits
* Total Accounts
* Total Offices

Shown as crisp business metrics.

### ✅ **2. Data Preview**

Random sample of 10 rows for instant validation.

### ✅ **3. Descriptive Statistics**

* Stats for numeric variables
* Frequency distribution of categorical variables

### ✅ **4. Geographic Insights**

Interactive Plotly bar chart showing:

> **Top 20 states by total deposit amount**

### ✅ **5. Modular Architecture**

The sidebar structure supports additional pages, such as:

* Exploratory Data Analysis
* Segmentation
* Advanced Analytics
* Predictive Modeling

These can be plugged in easily as the app scales.

---

## 📁 **Project Structure**

```
├── app.py
├── requirements.txt
├── populationgroup-wise-deposits.csv
└── cleaned_populationgroup_deposits.csv
```

---

## 🔧 **Installation & Setup**

### **1. Clone the repository**

```bash
git clone <https://github.com/Rugwed01/DS_Practical>
cd <project-folder>
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit app**

```bash
streamlit run app.py
```

### **4. Ensure the dataset files are in the same directory as `app.py`**

* `populationgroup-wise-deposits.csv`
* `cleaned-populationgroup-wise-deposits.csv`

---

## 🧠 **Tech Stack**

| Component        | Technology                  |
| ---------------- | --------------------------- |
| UI Framework     | Streamlit                   |
| Data Handling    | Pandas                      |
| Visualization    | Plotly, Matplotlib, Seaborn |
| ML/Stats Support | Scikit-Learn, Statsmodels   |
| Dashboard Layout | Streamlit Wide Mode         |

---

## 📊 **Example Visual (From Application)**

* Interactive bar chart of state-wise deposits
* KPIs showing aggregated business metrics
* Value-count tables for categorical segmentation

(All generated within the UI)

---

## 🏗️ **How It Works (Workflow)**

1. **Load Data:**
   Cached using `@st.cache_data` for performance.

2. **Clean Data:**
   Drop unused codes, convert year to category, remove all-zero entries.

3. **Compute KPIs:**
   Aggregation of deposits, accounts, and offices.

4. **Render Dashboard:**
   Display metrics → statistics → previews → charts.

5. **Sidebar Navigation:**
   Ready for multi-page expansion into EDA, ML modeling, segmentation, and forecasting.

---

## 📌 **Requirements**

All dependencies are listed in `requirements.txt`:

```
streamlit
pandas
plotly
seaborn
matplotlib
scikit-learn
statsmodels
```

---

## 📝 **Future Enhancements (Optional)**

Although not included directly in the current build, the architecture supports:

* Forecasting deposit trends using ML
* Clustering states/districts
* Geo-maps for spatial analysis
* Automated insights generation

---

## 👨‍💻 **Maintainer**

Rugwed Yawalkar

---
