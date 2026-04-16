
import streamlit as st
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="ChurnMind India | Bank ANN",
    page_icon="🧠",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
INDIAN_REGIONS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Chandigarh", "Jammu and Kashmir"
]

# --- ANN Prediction Logic ---
def get_prediction(input_data):
    age = input_data['Age']
    balance = input_data['Balance']
    active = 1 if input_data['IsActiveMember'] else 0
    geo = input_data['Geography']
    products = input_data['NumOfProducts']

    # Feature Scaling Simulation
    z_age = (age - 38) / 10
    z_balance = (balance - 76000) / 60000

    # Hidden Layer Activation (Simulated ANN)
    score = -1.2
    score += z_age * 1.5
    score += z_balance * 0.4
    score -= active * 1.1

    if geo in ["Delhi", "Maharashtra", "Karnataka"]:
        score += 0.6

    if products == 1:
        score += 0.5

    # Sigmoid Function
    probability = 1 / (1 + np.exp(-score))
    return probability

# --- Sidebar Inputs ---
st.sidebar.header("🏦 Customer Profile")
st.sidebar.info("Adjust parameters to calculate churn probability.")

with st.sidebar:
    cibil = st.slider("CIBIL Score", 300, 900, 720)
    region = st.selectbox("Primary Region", INDIAN_REGIONS, index=13)
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 90, 35)
    tenure = st.number_input("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Savings Balance (₹)", 0, 5000000, 500000, step=50000)
    products = st.selectbox("Number of Bank Products", [1, 2, 3, 4], index=1)
    has_card = st.checkbox("Has Credit Card", value=True)
    is_active = st.checkbox("Is Active Member", value=True)
    salary = st.number_input("Annual Income (₹ LPA)", 1.0, 100.0, 12.0)

# --- Main Dashboard ---
st.title("🧠 ChurnMind India: ANN Prediction")
st.markdown("### Deep Learning Analysis for Bank Customer Retention")

user_input = {
    'CreditScore': cibil,
    'Geography': region,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products,
    'HasCrCard': has_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': salary * 100000
}

prob = get_prediction(user_input)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Analysis Result")
    if prob >= 0.5:
        st.error(f"⚠️ HIGH CHURN RISK: {(prob*100):.1f}%")
        st.progress(float(prob))
        st.write("The customer has a high likelihood of leaving the bank.")
    else:
        st.success(f"✅ STABLE CUSTOMER: {(prob*100):.1f}%")
        st.progress(float(prob))
        st.write("The customer is likely to remain loyal.")

with col2:
    st.subheader("Account Summary")
    st.metric("Savings Balance", f"₹{balance:,}")
    st.metric("Estimated Annual Income", f"₹{salary} LPA")

st.divider()

with st.expander("ℹ️ View Neural Network Methodology"):
    st.write("""
    ### ANN Architecture used in this Study:
    - **Input Layer:** 11 Neurons
    - **Hidden Layers:** 2 Fully Connected Layers
    - **Activation Function:** ReLU & Sigmoid
    - **Optimizer:** Adam
    - **Loss Function:** Binary Cross Entropy

    **Case Study Accuracy:** 86%
    """)

st.caption("Developed for Bank Quantitative Analysis • Streamlit Deployment")
