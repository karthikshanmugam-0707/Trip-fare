import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import time

#loading the model
with open('gb_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('feature_columns.pkl', 'rb') as file:
    feature_columns = pickle.load(file)

st.title(':blue[🚖 Trip Fare Prediction]')
st.write(":red[Enter trip details]")

# Input for trip details
trip_distance = st.number_input("Trip Distance", min_value=0.0, step=0.1)
am_pm = st.selectbox("Time of Day", ["AM", "PM"])
am_pm = 0 if am_pm == "AM" else 1
is_night = st.selectbox("Is it a night trip?", ["Yes", "No"])
is_night = 1 if is_night == "Yes" else 0
payment_type = st.selectbox("Payment Type", ["gpay", "cash"])
payment_map = {
    "gpay": 1,
    "cash": 2
}
payment_type = payment_map[payment_type]
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, step=1)
trip_duration = st.number_input("Trip Duration", min_value=1, step=1)
hour = st.number_input("Hour", min_value=0, max_value=23, step=1)
is_rush_hour = st.selectbox("Is it rush hour?", ["Yes", "No"])
is_rush_hour = 1 if is_rush_hour == "Yes" else 0

input_data = {
    'trip_distance': trip_distance,
    'am/pm': am_pm,
    'is_night': is_night,
    'payment_type': payment_type,
    'passenger_count': passenger_count,
    'trip_duration': trip_duration,
    'hour': hour,
    'is_rush_hour': is_rush_hour
}
            

if st.button("submit"):
    with st.spinner("Wait for it..."):
        time.sleep(2)
        st.success("Done!")
       
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure the input DataFrame has the same columns as the model expects
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display the prediction
    st.write(f"Predicted Trip Fare: ${prediction[0]:.2f}")

sentiment_mapping = ["one", "two", "three", "four", "five"]
st.write(":green[Rate your experience with the trip fare prediction:]")
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"Thank you for {sentiment_mapping[selected]} star(s).")

       
