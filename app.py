import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CURRENT APP FOLDER
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# FIND DATASET
# Your screenshot shows the dataset name as:
# "Palo Alto Networks"
# Windows may be hiding the .csv extension.
# ============================================================
possible_files = [
    BASE_DIR / "Palo Alto Networks.csv",
    BASE_DIR / "Palo Alto Networks.xls",
    BASE_DIR / "Palo Alto Networks.xlsx",
    BASE_DIR / "Palo Alto Networks1.csv",
    BASE_DIR / "Palo Alto Networks1.xls",
    BASE_DIR / "Palo Alto Networks1.xlsx",
]

dataset_path = next((p for p in possible_files if p.exists()), None)

# ============================================================
# LOAD DATASET
# ============================================================
if dataset_path is None:
    st.error("❌ Dataset not found.")
    st.info(
        "Please keep your dataset in the same folder as app.py. "
        "Expected name: 'Palo Alto Networks.csv' "
        "(or .xls / .xlsx)."
    )

    st.write("📁 App folder:")
    st.code(str(BASE_DIR))

    st.write("Files currently found in this folder:")
    for file in BASE_DIR.iterdir():
        st.write("•", file.name)

    st.stop()

try:
    extension = dataset_path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(dataset_path)
    elif extension in [".xls", ".xlsx"]:
        df = pd.read_excel(dataset_path)
    else:
        st.error("Unsupported dataset format.")
        st.stop()

except Exception as e:
    st.error(f"❌ Error while loading dataset: {e}")
    st.stop()

# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================
model_path = BASE_DIR / "attrition_model.pkl"
preprocessor_path = BASE_DIR / "preprocessor.pkl"

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Could not load attrition_model.pkl: {e}")
    st.stop()

try:
    preprocessor = joblib.load(preprocessor_path)
except Exception as e:
    st.error(f"❌ Could not load preprocessor.pkl: {e}")
    st.stop()

# ============================================================
# TITLE
# ============================================================
st.title("📊 Employee Attrition Risk Prediction Dashboard")
st.subheader(
    "Machine Learning–Based Employee Attrition Prediction and Risk Scoring System"
)

st.success(f"✅ Dataset loaded: {dataset_path.name}")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Dashboard Filters")

# Department filter
department_column = None

for col in df.columns:
    if col.lower().strip() in ["department", "dept"]:
        department_column = col
        break

if department_column:
    departments = ["All"] + sorted(
        df[department_column].dropna().astype(str).unique().tolist()
    )

    selected_department = st.sidebar.selectbox(
        "Select Department",
        departments
    )

    if selected_department != "All":
        display_df = df[
            df[department_column].astype(str) == selected_department
        ].copy()
    else:
        display_df = df.copy()
else:
    selected_department = "All"
    display_df = df.copy()

# ============================================================
# DATASET INFORMATION
# ============================================================
st.markdown("### 📌 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Employees", len(display_df))

with col2:
    st.metric("Total Features", len(display_df.columns))

with col3:
    st.metric("Selected Department", selected_department)

# ============================================================
# SHOW DATA
# ============================================================
st.markdown("### 👥 Employee Dataset")
st.dataframe(display_df, use_container_width=True)

# ============================================================
# ATTRITION COLUMN
# ============================================================
attrition_column = None

possible_attrition_names = [
    "Attrition",
    "attrition",
    "Left",
    "left",
    "Employee_Attrition"
]

for col in possible_attrition_names:
    if col in df.columns:
        attrition_column = col
        break

# ============================================================
# ATTRITION SUMMARY
# ============================================================
if attrition_column:
    st.markdown("### 📈 Attrition Summary")

    attrition_counts = display_df[attrition_column].value_counts()

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Attrition Distribution**")
        st.bar_chart(attrition_counts)

    with c2:
        st.write("**Attrition Counts**")
        st.dataframe(
            attrition_counts.rename("Employees"),
            use_container_width=True
        )

# ============================================================
# MODEL INFORMATION
# ============================================================
st.markdown("### 🤖 Model Information")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Model", "Random Forest")

with m2:
    st.metric("Risk Levels", "Low / Medium / High")

with m3:
    st.metric("Prediction Type", "Employee Attrition")

# ============================================================
# RISK LEVEL INFORMATION
# ============================================================
st.markdown("### ⚠️ Risk Scoring")

st.write(
    """
    **Risk thresholds used in the project:**

    - 🟢 **Low Risk:** Probability < 0.30
    - 🟡 **Medium Risk:** Probability 0.30 – 0.60
    - 🔴 **High Risk:** Probability > 0.60
    """
)

# ============================================================
# EMPLOYEE PREDICTION SECTION
# ============================================================
st.markdown("### 🔮 Employee Attrition Prediction")

st.write(
    "The trained Random Forest model and preprocessor are loaded successfully. "
    "Use the dataset above for analysis and risk assessment."
)

st.info(
    "✅ Dashboard is connected to the trained model, preprocessor, and dataset."
)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Employee Attrition Risk Prediction Dashboard | Machine Learning Project")
