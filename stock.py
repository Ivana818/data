import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

stock_name=input('Enter the (US) stock code: ')
start_date=input('Enter the starting date (yyyy-mm-dd): ')
end_date=input('Enter the ending date (yyyy-mm-dd): ')
df=web.DataReader(stock_name, data_source='yahoo', start=start_date, end=end_date)
tday_pre=int(input('Enter the number of trading days that taking as reference to predict: '))

data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*.8)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

train_data=scaled_data[0:training_data_len, :]
x_train= []
y_train= []
for i in range(tday_pre, len(train_data)):
    x_train.append(train_data[i-tday_pre:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data=scaled_data[training_data_len-tday_pre: , :]
x_test=[]
y_test=dataset[training_data_len:, :]
for i in range(tday_pre, len(test_data)):
    x_test.append(test_data[i-tday_pre:i, 0])
    
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)


train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
plt.figure(figsize=(12,6))
plt.title('Prediction Model')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show