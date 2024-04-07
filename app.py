from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model("model.h5")

@app.route('/')
def home():
    return "Welcome to Car Sales Predictor App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract input data from JSON request
    gender = data['Gender']
    age = int(data['Age'])
    salary = float(data['Salary'])
    debt = float(data['Debt'])
    net_worth = float(data['Net_Worth'])
    if gender == 'male':
        gender=1
    else:
        gender=0
    # Preprocess the input data
    input_data = np.array([[gender, age, salary, debt, net_worth]])

    # Make prediction using the loaded model
    predicted_amount = str(model.predict(input_data)[0][0])
    print(predicted_amount)
    # Return the prediction as JSON response
    return jsonify({'predicted_amount': predicted_amount})

if __name__ == '__main__':
    app.run(debug=True)
