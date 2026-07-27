import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Load Model
# -------------------------
model = joblib.load("model/wine_model.pkl")
# Load dataset
df = pd.read_csv("dataset/Wine.csv")

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="AI Wine Customer Segment Prediction",
    page_icon="🍷",
    layout="wide"
)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("🍷 Wine Classifier")

st.sidebar.success("✅ Model Loaded Successfully")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Machine Learning Model")
st.sidebar.write("**Algorithm:** Random Forest Classifier")

st.sidebar.subheader("📊 Dataset Information")
st.sidebar.write("📌 Total Records: **178**")
st.sidebar.write("📌 Features: **13**")
st.sidebar.write("📌 Target Classes: **3**")

st.sidebar.markdown("---")

st.sidebar.subheader("🛠️ Technologies Used")

st.sidebar.write("• Python")
st.sidebar.write("• Pandas")
st.sidebar.write("• Scikit-learn")
st.sidebar.write("• Streamlit")
st.sidebar.write("• Joblib")

st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Project Objective")

st.sidebar.info(
    "Predict the customer segment of a wine based on its chemical properties using Machine Learning."
)

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Akshit Jamwal")

# -------------------------
# Main Title
# -------------------------
st.title("🍷 AI Wine Customer Segment Prediction System")


# -------------------------
# Input Fields
# -------------------------
col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol", value=13.0)
    malic_acid = st.number_input("Malic Acid", value=2.0)
    ash = st.number_input("Ash", value=2.3)
    alcalinity = st.number_input("Alcalinity of Ash", value=20.0)
    magnesium = st.number_input("Magnesium", value=100)
    total_phenols = st.number_input("Total Phenols", value=2.0)
    flavanoids = st.number_input("Flavanoids", value=2.0)

with col2:
    nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", value=0.3)
    proanthocyanins = st.number_input("Proanthocyanins", value=1.5)
    color_intensity = st.number_input("Color Intensity", value=5.0)
    hue = st.number_input("Hue", value=1.0)
    od280 = st.number_input("OD280/OD315", value=3.0)
    proline = st.number_input("Proline", value=1000)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Wine Segment"):

    input_data = pd.DataFrame(
        [[
            alcohol,
            malic_acid,
            ash,
            alcalinity,
            magnesium,
            total_phenols,
            flavanoids,
            nonflavanoid_phenols,
            proanthocyanins,
            color_intensity,
            hue,
            od280,
            proline
        ]],
        columns=[
            "Alcohol",
            "Malic_Acid",
            "Ash",
            "Ash_Alcanity",
            "Magnesium",
            "Total_Phenols",
            "Flavanoids",
            "Nonflavanoid_Phenols",
            "Proanthocyanins",
            "Color_Intensity",
            "Hue",
            "OD280",
            "Proline"
        ]
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)
    confidence = probabilities.max() * 100

    segment_names = {
    1: "🍇 Premium Wine",
    2: "🍷 Classic Wine",
    3: "🥂 Standard Wine"
}

st.success(f"### Prediction: {segment_names[prediction]}")

st.metric(
    label="Prediction Confidence",
    value=f"{confidence:.2f}%"
)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Developed using Python, Scikit-learn, Pandas and Streamlit")
# -------------------------
# Feature Importance Chart
# -------------------------

st.markdown("---")
st.subheader("📈 Feature Importance")

feature_names = [
    "Alcohol",
    "Malic Acid",
    "Ash",
    "Alcalinity",
    "Magnesium",
    "Total Phenols",
    "Flavanoids",
    "Nonflavanoid Phenols",
    "Proanthocyanins",
    "Color Intensity",
    "Hue",
    "OD280",
    "Proline"
]

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.invert_yaxis()

st.pyplot(fig)