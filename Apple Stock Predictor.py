# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:56:22 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:57:36 2020

@author: User
"""
#Installing required pakagess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas_datareader as web


#Dataframe
#df = pd.read_csv(r"D:\T\ML\NSE-TATAGLOBAL11.csv") #read

 
# Get the data for the stock Apple by specifying the stock ticker, start date, and end date 
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-12-28') 

new_df = df.filter(['Close'])

#creating train and test datasets
dataset = new_df.values
train = new_df.values[:1809, :]
test = new_df.values[1809:, :]

#x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#LSTM network with 2 layers of LSTM (50 neurons each) & 2 Dense layers (25 in one, 1 in the other)
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dense(units = 25))
model.add(Dense(units = 1))

#compiling the model
model.compile(optimizer = "adam", loss = "mean_squared_error")

#Training the model
model.fit(x_train, y_train, batch_size = 1, epochs = 2)

#creating test dataset
inputs = dataset[len(dataset) - len(test) - 60 : ]
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []

for i in range(60,inputs.shape[0]):
    x_test.append(inputs[i-60 : i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Getting the models predicted price values
closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)

#rmse(root mean sqaure error)
rms = np.sqrt(np.mean(np.power(test - closing_price, 2)))
print(rms)

#plotting the graph
train = new_df[ : 1809]
test = new_df[1809 : ]
test["Prediction"] = closing_price

plt.plot(train["Close"])
plt.plot(test[["Close", "Prediction"]])

