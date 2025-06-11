import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

st.title("Plant Growth Prediction using Logistic Regression")

st.sidebar.header("Input Parameters")

def user_input_features():
    Sunlight_Hours = st.sidebar.number_input('Sunlight Hours', min_value=0, max_value=24, value=12)
    Temperature = st.sidebar.number_input('Temperature (°C)', min_value=-10, max_value=50, value=25)
    Humidity = st.sidebar.number_input('Humidity (%)', min_value=0, max_value=100, value=50)
    Water_Adequacy = st.sidebar.number_input('Water Adequacy (0 or 1)', min_value=0, max_value=1, value=1)
    Fertilizer_Count = st.sidebar.number_input('Fertilizer Count', min_value=0, max_value=10, value=2)
    Stress_Index = st.sidebar.number_input('Stress Index (0 or 1)', min_value=0, max_value=1, value=0)
    Soil_Type_clay = st.sidebar.number_input('Soil Type Clay (0 or 1)', min_value=0, max_value=1, value=0)
    Soil_Type_loam = st.sidebar.number_input('Soil Type Loam (0 or 1)', min_value=0, max_value=1, value=1)
    Soil_Type_sandy = st.sidebar.number_input('Soil Type Sandy (0 or 1)', min_value=0, max_value=1, value=0)
    Water_Frequency_bi_weekly = st.sidebar.number_input('Water Frequency Bi-weekly (0 or 1)', min_value=0, max_value=1, value=0)
    Water_Frequency_daily = st.sidebar.number_input('Water Frequency Daily (0 or 1)', min_value=0, max_value=1, value=1)
    Water_Frequency_weekly = st.sidebar.number_input('Water Frequency Weekly (0 or 1)', min_value=0, max_value=1, value=0)
    Fertilizer_Type_chemical = st.sidebar.number_input('Fertilizer Type Chemical (0 or 1)', min_value=0, max_value=1, value=1)

    data = {
        'Sunlight_Hours': Sunlight_Hours,
        'Temperature': Temperature,
        'Humidity': Humidity,
        'Water_Adequacy': Water_Adequacy,
        'Fertilizer_Count': Fertilizer_Count,
        'Stress_Index': Stress_Index,
        'Soil_Type_clay': Soil_Type_clay,
        'Soil_Type_loam': Soil_Type_loam,
        'Soil_Type_sandy': Soil_Type_sandy,
        'Water_Frequency_bi-weekly': Water_Frequency_bi_weekly,
        'Water_Frequency_daily': Water_Frequency_daily,
        'Water_Frequency_weekly': Water_Frequency_weekly,
        'Fertilizer_Type_chemical': Fertilizer_Type_chemical
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

model_filename = "logistic_regression_model_plant_health.pkl"  
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

if st.button('Predict Growth Milestone'):
    prediction = model.predict(input_df)

    st.subheader('Prediction for User Input')
    st.write(f"Predicted Growth Milestone: {prediction[0]}")

