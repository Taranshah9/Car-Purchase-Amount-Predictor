import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('D:\Car Sales Predictor\CarSalesPredictor\Car_Purchasing_Data.csv', encoding='ISO-8859-1')
print(df.head(5))

# sns.pairplot(df)

df = df.drop(['Customer Name','Customer e-mail'],axis=1)
print(df)

y = pd.DataFrame(df['Car Purchase Amount'])
print(y)

df = df.drop(['Car Purchase Amount','Country'],axis=1)
print(df)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

y_scaled = scaler.fit_transform(y)
print(y_scaled)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size=0.3,random_state=42)

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(40,input_dim = 5,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))

print(model.summary())

model.compile(optimizer = 'adam',loss = 'mean_squared_error')

epochs_hist = model.fit(X_train,y_train,epochs=100,batch_size = 25,verbose = 1,validation_split = 0.2)

# plt.plot(epochs_hist.history['loss'])
# plt.plot(epochs_hist.history['val_loss'])
# plt.title("Model Loss during training")
# plt.ylabel("Training and Validation Loss")
# plt.xlabel("Epochs")
# plt.legend(['Training Loss','Validation Loss'])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Gender = input("Enter Gender")
if Gender == 'male':
  Gender=1
else:
  Gender=0
# Age = int(input('Enter Age:'))
# salary = float(input('Enter Annual Salary:'))
# debt = float(input('Enter Credit card debt:'))
# net_worth = float(input('Enter net worth:'))

# X_example = np.array([[Gender,Age,salary,debt,net_worth]])
# estimated_cost = model.predict(X_example)
# print("Estimated Purchase Amount:",estimated_cost)

model.save("model.h5")
from keras.models import load_model
loaded_model = load_model("model.h5")
