import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras import optimizers, regularizers
import matplotlib.pyplot as plt
import math
#X = np.array
import csv

    
Input_Data = np.genfromtxt('logged_data.csv', delimiter=',')
Output_Data_UTM = np.genfromtxt('logged_data.csv', delimiter=',')


Input_Data_Second = np.genfromtxt('logged_data_trial2.csv', delimiter=',')
Output_Data_UTM_Second = Input_Data_Second[:,:2]
Output_Data_UTM_Second = Output_Data_UTM_Second[1:,:]
BottomLeftX = Output_Data_UTM_Second[:,0].min()
BottomLeftY = Output_Data_UTM_Second[:,1].min()

Output_Data_UTM_Second[:,0] = Output_Data_UTM_Second[:,0] - BottomLeftX  ##set reference point to minimum x and y values
Output_Data_UTM_Second[:,1] = Output_Data_UTM_Second   [:,1] - BottomLeftY



Input_Data_Second = np.delete(Input_Data_Second,[0],axis=0)
Input_Data_Second = Input_Data_Second[:,6:]
Input_Data_Second = np.delete(Input_Data_Second, [2,3,4,5,6],axis = 1)  ##removing all unnecessary data from 2 camera data file
Input_Data_Second = Input_Data_Second[:,:4]




Output_Data_UTM = np.delete(Input_Data,[0,1,2], axis=0)
Output_Data_UTM = np.delete(Output_Data_UTM, list(range(2,Output_Data_UTM.shape[1])),  axis = 1)

print(Output_Data_UTM.shape)

#Removing extra columns in csv

Input_Data = np.delete(Input_Data,[0,1,2], axis=0)
Input_Data = np.delete(Input_Data,list(range(6)),axis=1)
Input_Data = np.delete(Input_Data,[3,4,5,6],axis=1)
Input_Data = np.delete(Input_Data,[6,7,8,9],axis=1)
Input_Data = np.delete(Input_Data,[9,10,11,12],axis=1)
Input_Data = np.delete(Input_Data,[2,5,8,11],axis=1)

numSamples = Input_Data.shape[0]


print(Output_Data_UTM.shape)

#Retrieving and subtracting smallest X and Y values of UTM coordinates
BottomLeftX = Output_Data_UTM[:,0].min()
BottomLeftY = Output_Data_UTM[:,1].min()

Output_Data_UTM[:,0] = Output_Data_UTM[:,0] - BottomLeftX  ##set reference point to minimum x and y values
Output_Data_UTM[:,1] = Output_Data_UTM[:,1] - BottomLeftY
averages = np.mean(Output_Data_UTM, axis=0)

#Feature vector: x, y, z of camera, x, y of center pixel of object, area of object, declination angle
hiddenLayers = 3 # Simple input to output mapping, no hidden layers if 0
hiddenLayerDimension = 10
#Y_Vector_Size = 2  # Predicting x, y, and z UTM coordinates of the object


totalSamples = Input_Data_Second.shape[0]
lossList = []
lossListVal = []
Input_DataSubset = np.empty((0,Input_Data.shape[1]))
Output_Data_UTMSubset = np.empty((0,Output_Data_UTM.shape[1]))
iterations = 50
subsetLength = int(totalSamples/iterations)
nums=np.arange(len(Input_Data))
num_Samples = int(totalSamples*(1/iterations))
print("NumSamples = ", num_Samples)
'''
for i in range(iterations): ## 20 iterations
    print("Iteration number = ", i)
    choices = np.random.choice(nums, num_Samples,replace=False) ##Generate certain number of unique random sample indices to choose from
    #print("Choices = ", choices)   
    print("Choices length = ", len(choices))
    Input_DataSubset = np.append(Input_DataSubset, Input_Data[choices,:], axis=0)  # add to current data subset
    print("Data Subset Length = ", len(Input_DataSubset))
    print("Remaining input data shape = ", Input_Data.shape)
    print("Input data choices about to be deleted: ", Input_Data[choices,:])
    Input_Data = np.delete(Input_Data, choices,axis=0) #remove data subset from total dataset
    print("Remaining input data shape after deletion: ", Input_Data.shape)

    #print("Subset shape: ", Input_DataSubset.shape)
    #print("Total data shape: ", Input_Data.shape)
    Output_Data_UTMSubset = np.append(Output_Data_UTMSubset, Output_Data_UTM[choices,:], axis=0)
    Output_Data_UTM = np.delete(Output_Data_UTM, choices,axis=0)
    #print("Output Data length: ", Output_Data_UTMSubset.shape[0])
    
    nums = np.delete(nums, choices)
    
    nums = np.arange(len(nums))
'''    
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

hiddenLayerDimension = 13
model = Sequential()
model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=Input_Data_Second.shape[1]))  ##input layer
#model.add(Dropout(0.25)) 
for i in range(3):
    model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=hiddenLayerDimension))##hidden layers

#model.add(Dropout(0.25))

#model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=hiddenLayerDimension))
#model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=hiddenLayerDimension)) 
#model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=hiddenLayerDimension)) 
model.add(Dense(units=hiddenLayerDimension, activation='relu', kernel_regularizer=regularizers.l2(3),input_dim=hiddenLayerDimension)) ##output layers

#model.add(Dropout(0.25))
model.add(Dense(units=Output_Data_UTM_Second.shape[1], activation='linear', input_dim=hiddenLayerDimension))
adam=optimizers.Adam(lr=0.0007, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mse', optimizer=adam, metrics=['mse'])

#history = model.fit(Input_Data[:int(.6*samples)], Output_Data_UTM[:int(.6*samples)], epochs=50,batch_size = 10, verbose = 2)
history = model.fit(Input_Data_Second   , Output_Data_UTM_Second, validation_split=0.1,epochs=100,batch_size =20, verbose = 2)
#history = model.fit(Input_DataSubset, Output_Data_UTMSubset, validation_split=0.2,epochs=50,batch_size = 20, verbose = 2)
preds = model.predict(Input_Data_Second[int(.8*totalSamples):,:])
diffs = preds - Output_Data_UTM_Second  [int(0.8*totalSamples):,:]
avgStraightLineDist = np.mean(np.sqrt(np.sum(diffs[:,:]**2, axis = 1)))
#loss = np.array(history.history['acc'])
#val_loss = np.array(history.history['val_acc'])
#straightLineDistList.append(avgStraightLineDist)
#loss = loss[-1]
#val_loss = np.sqrt(val_loss)
#val_loss = val_loss[-1]
#lossList.append(loss)
#lossListVal.append(val_loss)
print("Average straight line distance is: ", avgStraightLineDist)
print(history.history.keys())
#plt.plot(preds)
#plt.plot(np.sqrt(history.history['loss']))
#plt.plot(np.sqrt(history.history['val_loss']))
#plt.yscale('log')
#plt.xlim(0,100)
#plt.title('Mean Squared Error Loss at Each Epoch')
#plt.ylabel('Straight Line Distance Error (meters)')
#plt.xlabel('Epoch Number')
#plt.legend(['train', 'test'], loc='upper right')
#plt.xlim(0,9)
#plt.show()

'''


Testresults = np.abs(preds - Output_Data_UTMSubset[int(.8*totalSamples):,:])
print(Testresults)
StraightLineDist = np.sqrt((Testresults[:,0])**2 + (Testresults[:,1])**2)
avgStraightLineDist = np.sum(StraightLineDist)/len(StraightLineDist)
print(avgStraightLineDist)


def mse(predictions, actual):
    print(len(predictions))
    return sum((predictions- actual)**2)/len(predictions)

#first = mse(preds[:,0],Output_Data_UTM[int(.8*totalSamples):,0])
#second = mse(preds[:,1],Output_Data_UTM[int(.8*totalSamples):,1])

print(first, second)

# Plot training & validation accuracy values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''