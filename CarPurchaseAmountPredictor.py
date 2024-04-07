import streamlit as st
import requests

def get_prediction(input_data):
    try:
        # Send a POST request to Flask app to get prediction
        response = requests.post("http://localhost:5000/predict", json=input_data)
        
        # Check if the request was successful
        if response.ok:
            prediction = response.json().get('predicted_amount')
            if prediction is not None:
                return prediction
            else:
                return "Prediction not available"
        else:
            return "Failed to fetch prediction"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Streamlit UI
st.subheader('Car Purchase Amount Predictor')
st.title('Welcome to the site!!')
st.write("To make a prediction about a customer's possible purchase  amount, please enter their data below.")
# Input form
gender = st.text_input('Gender(male/female)')
age = st.text_input('Age')
salary = st.text_input('Salary')
debt = st.text_input('Debt')
net_worth = st.text_input('Net Worth')

input_data = {
    'Gender': gender,
    'Age': age,
    'Salary': salary,
    'Debt': debt,
    'Net_Worth': net_worth
}

if st.button('Predict'):
    with st.spinner('Predicting...'):
        prediction = get_prediction(input_data)
        st.success(f'Predicted amount: {prediction}')
