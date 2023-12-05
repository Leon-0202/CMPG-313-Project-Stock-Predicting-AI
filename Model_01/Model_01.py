# Makes predictions using the last 60 values of the training data

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error



# Importing training data
training_file = 'Data_Netflix.csv'
dataset_train = pd.read_csv(training_file)
training_set = dataset_train.iloc[:, 1:2].values



# Data Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)



# Incorporating Timesteps Into Training Data
timestep_size = 60
X_train = []
y_train = []
for i in range(timestep_size, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timestep_size:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    


# Creating the LSTM Model
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=10,batch_size=32)



# Importing test data
testing_file = 'Data_Apple.csv'
dataset_test = pd.read_csv(testing_file)
real_stock_price = dataset_test.iloc[:, 1:2].values



# Incorporating Timesteps Into Testing Data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestep_size:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(timestep_size, len(inputs)):
    X_test.append(inputs[i-timestep_size:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# Making predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)



# Plotting the Results
plt.plot(real_stock_price, color = 'black', label = 'Actual Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend()
plt.show()



# Calculating the Error
rmse = mean_squared_error(real_stock_price, predicted_stock_price, squared = False)
mape = mean_absolute_percentage_error(real_stock_price, predicted_stock_price)
print("RSME: ", rmse)
print("MAPE: ", mape)