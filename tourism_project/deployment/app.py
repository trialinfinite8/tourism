import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="trialinfinite8/tourism-package-prediction", filename="best_purchase_preiction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("The Tourism Package Purchase Prediction App is an internal tool for sales staff that predicts whether a customer is likely to purchase a tourism package based on their demographic and engagement details.")
st.write("Kindly enter the customer details to check whether they are likely to convert.")

# Collect user input
TypeofContact = st.selectbox("Type of Contact (how the customer was contacted)", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender (customer's gender)", ["Female", "Male"])
ProductPitched = st.selectbox("Product Pitched (type of product pitched to the customer)", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status (customer's marital status)", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation (customer's job designation)", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
DurationOfPitch = st.number_input("Duration of Pitch (duration of sales pitch in minutes)", min_value=5, max_value=120, value=15)
MonthlyIncome = st.number_input("Monthly Income (customer's gross monthly income)", min_value=1000, max_value=999999, value=20000)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (including the customer)", min_value=1, max_value=5, value=2)
NumberOfFollowups = st.number_input("Number of Followups (followups made after the pitch)", min_value=1, max_value=10, value=3)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
NumberOfTrips = st.number_input("Number of Trips (trips taken in a year)", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
CityTier = st.selectbox("City Tier (tier of the customer's city)", [1, 2, 3])
Passport = st.selectbox("Passport (does the customer have a passport?)", ["Yes", "No"])
OwnCar = st.selectbox("Own Car (does the customer own a car?)", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)

# Convert inputs to match model training
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'MonthlyIncome': MonthlyIncome,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'NumberOfTrips': NumberOfTrips,
    'PreferredPropertyStar': PreferredPropertyStar,
    'CityTier': CityTier,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore
}])

# Set the classification threshold
classification_threshold = 0.5

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "convert" if prediction == 1 else "not convert"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
