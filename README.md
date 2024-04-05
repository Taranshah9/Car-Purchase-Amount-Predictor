# Car Purchase Amount Prediction using Neural Network

This project aims to predict the car purchase amount using a neural network model. We utilize the Car Purchasing Data CSV file for this analysis. Here's a breakdown of what the code does:

## Core Concepts Covered
- **Data Loading and Exploration**: The project starts with loading the data from the CSV file using Pandas. We then explore the first few rows of the dataset to understand its structure.

- **Data Preprocessing**: Preprocessing steps include dropping unnecessary columns ('Customer Name' and 'Customer e-mail') and scaling the input features and target variable using Min-Max scaling.

- **Train-Test Split**: The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.

- **Neural Network Model Building**: We construct a feedforward neural network using Keras Sequential API. The model consists of several dense layers with ReLU activation functions and a linear output layer.

- **Model Training**: The model is trained using the training data with a specified number of epochs and batch size. We utilize mean squared error as the loss function and Adam optimizer for optimization.

- **Model Evaluation**: After training, the model is evaluated on the testing data. We calculate metrics such as Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R-squared to assess the model's performance.

## Files Included
- **Car_Purchasing_Data.csv**: Input dataset containing car purchasing data.
- **car_purchase_amount_prediction.ipynb**: Jupyter Notebook containing the Python code.
- **README.md**: Markdown file explaining the project and code.

## Dependencies
- pandas
- scikit-learn
- matplotlib
- seaborn
- TensorFlow
- Keras

## Usage
1. Clone the repository: `git clone https://github.com/your-username/car-purchase-prediction.git`
2. Navigate to the project directory: `cd car-purchase-prediction`
3. Run the Jupyter Notebook: `jupyter notebook car_purchase_amount_prediction.ipynb`
4. Follow the instructions in the notebook to execute the code and observe the results.

## Results
Upon running the code, you'll observe the training and validation loss plot, as well as the evaluation metrics for the model's performance on the test set.

## Conclusion
This project demonstrates the process of building and training a neural network model for predicting car purchase amounts. By leveraging the provided car purchasing data, we can develop insights into customer behavior and make informed decisions in marketing and sales strategies.
