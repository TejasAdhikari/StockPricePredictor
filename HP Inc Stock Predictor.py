# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 23:02:41 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:57:36 2020

@author: User
"""
#Installing required pakagess
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import yfinance as yf   
import pandas_datareader as web


#Dataframe
#df = pd.read_csv(r"D:\T\ML\NSE-TATAGLOBAL11.csv") #read

 
# Get the data for the stock Apple by specifying the stock ticker, start date, and end date 
df = web.DataReader('HPQ', data_source='yahoo', start='2012-01-01', end='2020-12-28') 

new_df = df.filter(['Close'])

#creating train and test datasets
dataset = new_df.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 

#x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]

#Split the data into x_train and y_train data sets
x_train, y_train = [], []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    
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
#Test data set
test_data = scaled_data[training_data_len - 60: , : ]
#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Getting the models predicted price values
closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)

#rmse(root mean sqaure error)
rms = np.sqrt(np.mean(np.power(y_test- closing_price, 2)))
print(rms)

#plotting the graph
train = new_df[:training_data_len]
valid = new_df[training_data_len:]
valid['Prediction'] = closing_price

plt.plot(train["Close"])
plt.plot(valid[["Close", "Prediction"]])






