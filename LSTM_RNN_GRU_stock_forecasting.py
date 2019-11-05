
# LSTM

import time

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.layers import Flatten
from keras.layers import SimpleRNN
from keras.layers import GRU


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import pandas_datareader.data as web
dataframe1 = web.DataReader('IBM',data_source="yahoo",start="01/01/2005",end="01/01/2019").dropna() 



dataframe = dataframe1.dropna()
dataframe = dataframe[['Open', 'High', 'Low', 'Close']]

dataframe = dataframe[['Close']]
first_price = dataframe.Close.iloc[0]
print(dataframe.shape, dataframe.head())

print(first_price)

print(dataframe.head())
dataframe.Close.plot()

# LSTM with sliding window

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1): 
		data = dataset[i:(i+look_back), 0]      
		dataX.append(data)
		dataY.append(dataset[i + look_back, 0]) 
	return numpy.array(dataX), numpy.array(dataY)


#fix random seed for reproducibility
#numpy.random.seed(42955)

# load the dataset
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset)+1,:]

# reshape 
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1)) # trainX.shape[1] = 10 = look_back/timesteps
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))  # testX.shape[1] = 10 = look_back/timesteps



model = Sequential()

model.add(LSTM(5, activation = 'tanh',inner_activation = 'hard_sigmoid', input_shape=(look_back,1),return_sequences=True))#, kernel_initializer='glorot_uniform'))

model.add(LSTM(1, activation = 'tanh',inner_activation = 'hard_sigmoid',return_sequences=False))

model.add(Dense(1, activation = 'relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
opt = Adagrad( lr = 0.001 )
adam_opt = Adam(lr =0.01)
opt2 = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)



model.compile(loss= tf.losses.huber_loss, optimizer= opt2, metrics = ['mae']) #tf.losses.huber_loss for huber loss

start = time.time()

history = model.fit(trainX, trainY, epochs = 400 , batch_size= 30, shuffle=False,  validation_split = 0.15)

print('training time in seconds : ', int(time.time() - start))

# -  - * * * - - ## -  - * * * - - ## -  - * * * - - ## -  - * * * - - #



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.3f RMSE' % (testScore))

#Plot

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# plot baseline and predictions
plt.style.use('ggplot')
plt.figure(figsize=(12,6), dpi=110)
plt.grid(color='grey', linestyle='dashed')
plt.xlabel('Observations')
plt.ylabel('IBM',rotation=90)
plt.plot(scaler.inverse_transform(dataset), label = 'Actual Closing Prices', linewidth = 1.2, color = 'c')
plt.plot(trainPredictPlot, label = 'A.I. Train Data Price Predictions_After fit', linewidth = 0.9, color = 'k')
plt.plot(testPredictPlot, label = 'A.I. Test Data Price Predictions', linewidth = 0.9, color = 'r')
legend = plt.legend(fontsize = 12,frameon = True)
legend.get_frame().set_edgecolor('b')
legend.get_frame().set_linewidth(0.4)

plt.show()







#RNN

import pandas_datareader.data as web
dataframe1 = web.DataReader('IBM',data_source="yahoo",start="01/01/2005",end="01/01/2019").dropna() 



dataframe = dataframe1.dropna()
dataframe = dataframe[['Open', 'High', 'Low', 'Close']]

dataframe = dataframe[['Close']]
first_price = dataframe.Close.iloc[0]
print(dataframe.shape, dataframe.head())

dataframe.Close.plot()



# RNN with sliding window

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1): 
		data = dataset[i:(i+look_back), 0]      
		dataX.append(data)
		dataY.append(dataset[i + look_back, 0]) 
	return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
#numpy.random.seed(115537)

# load the dataset
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset)+1,:]

# reshape 
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1)) # trainX.shape[1] = 10 = look_back/timesteps
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))  # testX.shape[1] = 10 = look_back/timesteps


model2 = Sequential()

model2.add(SimpleRNN(5, input_shape=(look_back,1),return_sequences=True)) #, kernel_initializer = 'glorot_normal'))
model2.add(SimpleRNN(1, return_sequences=False))


model2.add(Dense(1, activation = 'relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
opt = Adagrad( lr = 0.001 )
adam_opt = Adam(lr =0.01)
opt2 = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)



model2.compile(loss= tf.losses.huber_loss, optimizer= opt2, metrics = ['mae']) #tf.losses.huber_loss for huber loss

start = time.time()

history = model2.fit(trainX, trainY, epochs = 200 , batch_size= 30, shuffle=False,  validation_split = 0.15)

print('training time in seconds : ', int(time.time() - start))

# make predictions
trainPredict = model2.predict(trainX)
testPredict = model2.predict(testX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.3f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


#Plot
plt.style.use('ggplot')
plt.figure(figsize=(13,7), dpi=110)
plt.grid(color='grey', linestyle='dashed')
plt.xlabel('Observations')
plt.ylabel('IBM',rotation=90)
plt.plot(scaler.inverse_transform(dataset), label = 'Actual Closing Prices', linewidth = 1.2, color = 'c')
plt.plot(trainPredictPlot, label = 'A.I. Train Data Price Predictions_After fit', linewidth = 0.9, color = 'k')
plt.plot(testPredictPlot, label = 'A.I. Test Data Price Predictions', linewidth = 0.9, color = 'r')
legend = plt.legend(fontsize = 12,frameon = True)
legend.get_frame().set_edgecolor('b')
legend.get_frame().set_linewidth(0.4)

plt.show()







#GRU

import pandas_datareader.data as web
dataframe1 = web.DataReader('IBM',data_source="yahoo",start="01/01/2005",end="01/01/2019").dropna() 



dataframe = dataframe1.dropna()
dataframe = dataframe[['Open', 'High', 'Low', 'Close']]

dataframe = dataframe[['Close']]
first_price = dataframe.Close.iloc[0]
print(dataframe.shape, dataframe.head())

dataframe.Close.plot()




# GRU with sliding window

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1): 
		data = dataset[i:(i+look_back), 0]      
		dataX.append(data)
		dataY.append(dataset[i + look_back, 0]) 
	return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
#numpy.random.seed(1708131234)

# load the dataset
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset)+1,:]

# reshape 
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1)) # trainX.shape[1] = 10 = look_back/timesteps
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))  # testX.shape[1] = 10 = look_back/timesteps


model3 = Sequential()

model3.add(GRU(5, activation='tanh', recurrent_activation='sigmoid', input_shape=(look_back,1), return_sequences=True)) #, kernel_initializer='glorot_normal',))
model3.add(GRU(1, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)) #, kernel_initializer='uniform',))


model3.add(Dense(1, activation = 'relu'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
opt = Adagrad( lr = 0.001 )
adam_opt = Adam(lr =0.01)
opt2 = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)



model3.compile(loss= tf.losses.huber_loss, optimizer= opt2, metrics = ['mae']) #tf.losses.huber_loss for huber loss

start = time.time()

history = model3.fit(trainX, trainY, epochs = 200 , batch_size= 30, shuffle=False,  validation_split = 0.15)

print('training time in seconds : ', int(time.time() - start))


# -  - * * * - - ## -  - * * * - - ## -  - * * * - - ## -  - * * * - - #



# make predictions
trainPredict = model3.predict(trainX)
testPredict = model3.predict(testX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.3f RMSE' % (testScore))


#Plot
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


plt.style.use('ggplot')
plt.figure(figsize=(13,7), dpi=110)
plt.grid(color='grey', linestyle='dashed')
plt.xlabel('Observations')
plt.ylabel('IBM',rotation=90)
plt.plot(scaler.inverse_transform(dataset), label = 'Actual Closing Prices', linewidth = 1.2, color = 'c')
plt.plot(trainPredictPlot, label = 'A.I. Train Data Price Predictions_After fit', linewidth = 0.9, color = 'k')
plt.plot(testPredictPlot, label = 'A.I. Test Data Price Predictions', linewidth = 0.9, color = 'r')
legend = plt.legend(fontsize = 12,frameon = True)
legend.get_frame().set_edgecolor('b')
legend.get_frame().set_linewidth(0.4)

plt.show()




