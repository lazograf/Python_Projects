import time

import tensorflow as tf
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.initializers import RandomNormal

from keras.layers import Flatten


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pandas_datareader.data as web
dataframe1 = web.DataReader('IBM',data_source="yahoo",start="01/01/2005",end="01/01/2019").dropna()


dataframe = dataframe1.iloc[:,:5].dropna()
#dataframe = dataframe[['Open', 'High', 'Low', 'Close']]

column_reordering = ['High', 'Low', 'Open', 'Volume', 'Close']



#Pandas method to swap column positions inside the dataframe
dataframe = dataframe.reindex(columns=column_reordering)
print(dataframe.head())

print("\n", dataframe.describe())

dataframe_plot = dataframe

dataframe.Volume.plot()



dataframe.iloc[:,-2] = dataframe.iloc[:,-2] / 1e6
dataframe.Volume.plot()

dataframe.Close.plot()

# convert an array of values into a dataset matrix
def create_dataset(dataset,dataset2, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1): 
		data = dataset[i:(i+look_back), :4]      
		dataX.append(data)
		dataY.append(dataset2[i + look_back]) 
	return numpy.array(dataX), numpy.array(dataY)



# fix random seed for reproducibility
numpy.random.seed(7382582)


dataset = dataframe
#dataset = dataset.astype('float32')


dataset_features = dataset.iloc[:,:-1].values
dataset_target = dataset.iloc[:,-1].values

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_features = scaler.fit_transform(dataset_features)

scaler2 = MinMaxScaler(feature_range=(0,1))
dataset_target =  scaler2.fit_transform(dataset_target.reshape(-1,1))




# split into train and test sets
train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train_f, test_f = dataset_features[0:train_size+1,:], dataset_features[train_size+1:len(dataset)+1,:]
train_t, test_t = dataset_target[0:train_size+1], dataset_target[train_size+1:len(dataset)+1]


# reshape 
look_back = 10
trainX, trainY = create_dataset(train_f,train_t, look_back = look_back)
testX, testY = create_dataset(test_f,test_t, look_back = look_back)



print("\n", "training set ONE sample FEATURES", "\n", trainX[1,:], "\n", "\n", "training set ONE sample TARGET", testY[1])


# "Sanity check"
print(trainX.shape,trainY.shape)

trainX[:2,:] # (!) 3-D array: In order to train the sequential model, I pass as input shape the second and third dimension,
#namely the number of rows and columns



# MLP FFNN
model = Sequential()
model.add(Dense(600, input_shape = (trainX.shape[1],trainX.shape[2]), activation = 'relu', kernel_initializer='uniform'))
model.add(Dropout(0.4))
model.add(Dense(400, activation = 'relu', kernel_initializer='uniform' ))
model.add(Dropout(0.3))
model.add(Dense(150, activation = 'relu' ))
model.add(Dropout(0.1))
model.add(Dense(100, activation = 'hard_sigmoid' ))
model.add(Dense(50, activation = 'relu' ))
model.add(Dense(10, activation = 'relu' ))
model.add(Flatten())
model.add(Dense(1, activation = 'relu'))


opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Adagrad(lr=0.001)

model.compile(loss=tf.losses.huber_loss, optimizer= opt, metrics = ['mae'])



start = time.time()

history = model.fit(trainX, trainY, epochs = 500 , batch_size= 32, shuffle=True ,validation_split = 0.15)

print('training time : ', time.time() - start)


model.summary()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

                             
# invert predictions
trainPredict = scaler2.inverse_transform(trainPredict)
testPredict = scaler2.inverse_transform(testPredict)

from sklearn.metrics import mean_squared_error as MSE
import math

#MSE(y_true,y_pred)
train_accuracy = math.sqrt(MSE(scaler2.inverse_transform(trainY), trainPredict))

print('Root mean squared error on the train dataset: ', round(train_accuracy,3))

test_accuracy = math.sqrt(MSE(scaler2.inverse_transform(testY), testPredict))

print('Root mean squared error on the test dataset: ', round(test_accuracy,3))

trainY,testY = dataset_target[0:train_size+1], dataset_target[train_size+1:len(dataset)+1]



# Plot
trainPredictPlot = numpy.empty_like(dataset.Close.values)
trainPredictPlot[:] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.flatten()
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset.Close.values)
testPredictPlot[:] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict.flatten()



# plot baseline and predictions
plt.style.use('ggplot')
plt.figure(figsize=(12,6), dpi=110)
plt.grid(color='grey', linestyle='dashed')
plt.xlabel('Observations')
plt.ylabel('IBM',rotation=90)
plt.plot(dataset.Close.values, label = 'Actual Closing Prices', linewidth = 1.2, color = 'c')
plt.plot(trainPredictPlot, label = 'A.I. Train Data Price Predictions_After fit', linewidth = 0.9, color = 'k')
plt.plot(testPredictPlot, label = 'A.I. Test Data Price Predictions', linewidth = 0.9, color = 'r')
legend = plt.legend(fontsize = 12,frameon = True)
legend.get_frame().set_edgecolor('b')
legend.get_frame().set_linewidth(0.4)
plt.show()

