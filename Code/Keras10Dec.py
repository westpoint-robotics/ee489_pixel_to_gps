import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras import optimizers, regularizers,initializers
import matplotlib.pyplot as plt
import math
import csv

Input_Data = np.genfromtxt('columns_removed_logged_data_trial2.csv', delimiter=',')
Output_Data_UTM = np.genfromtxt('columns_removed_logged_data_trial2.csv', delimiter=',')

Input_Data = Input_Data[3:,2:] #remove top 3 rows and first two columns for input data
Input_Data = Input_Data[:,[3,4,5]]   ## use this line if you want to remove more columns (subset of cameras)
Output_Data_UTM = Output_Data_UTM[3:,:2]
print(Input_Data.shape, Output_Data_UTM.shape)


totalSamples = Input_Data.shape[0]  ##this section is for normalization of data where min x and min y are reference points  (0,0)

BottomLeftX = Input_Data[:,0].min()
BottomLeftY = Input_Data[:,1].min()
Input_Data[:,0] = Input_Data[:,0] - BottomLeftX  ##set reference point to minimum x and y values
Input_Data[:,1] = Input_Data[:,1] - BottomLeftY
'''
BottomLeftX = Input_Data[:,3].min()
BottomLeftY = Input_Data[:,4].min()
Input_Data[:,3] = Input_Data[:,3] - BottomLeftX  ##set reference point to minimum x and y values
Input_Data[:,4] = Input_Data[:,4] - BottomLeftY

BottomLeftX = Input_Data[:,6].min()
BottomLeftY = Input_Data[:,7].min()
Input_Data[:,6] = Input_Data[:,6] - BottomLeftX  ##set reference point to minimum x and y values
Input_Data[:,7] = Input_Data[:,7] - BottomLeftY

BottomLeftX = Input_Data[:,9].min()
BottomLeftY = Input_Data[:,10].min()
Input_Data[:,9] = Input_Data[:,9] - BottomLeftX  ##set reference point to minimum x and y values
Input_Data[:,10] = Input_Data[:,10] - BottomLeftY
'''
#Retrieving and subtracting smallest X and Y values of UTM coordinates
BottomLeftX = Output_Data_UTM[:,0].min()-5
BottomLeftY = Output_Data_UTM[:,1].min()-5
Output_Data_UTM[:,0] = Output_Data_UTM[:,0] - BottomLeftX  ##set reference point to minimum x and y values
Output_Data_UTM[:,1] = Output_Data_UTM[:,1] - BottomLeftY

def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


## this block picks a random subset of 90% of all samples for training data, other 10% is the test set.351*25
nums=np.arange(Input_Data.shape[0])
choices = np.random.choice(nums, int(totalSamples*0.9),replace=False)
Input_Data_sub = Input_Data[choices] ##subsets for training
Output_Data_UTM_sub = Output_Data_UTM[choices]
Input_Data_Test = np.delete(Input_Data, choices,axis=0)  ##testing subsets
Output_Data_UTM_Test = np.delete(Output_Data_UTM, choices,axis=0)
Input_Data_sub_iteration = np.empty((0,Input_Data.shape[1]))
Output_Data_UTM_sub_iteration = np.empty((0,Output_Data_UTM.shape[1]))
#
#moreNums = np.arange(Input_Data_sub.shape[0])
#moreChoices = np.random.choice(moreNums, int(.1*Input_Data_sub.shape[0]), replace=False)

#Input_Data_sub = Input_Data_sub[moreChoices]
#Output_Data_UTM_sub = Output_Data_UTM_sub[moreChoices]


out =[]
#for k in range(1,101):
'''
    nums = np.arange(Input_Data_sub.shape[0])
    choices = np.random.choice(nums, int(8802/100),replace=False)

    Input_Data_sub_iteration = np.vstack((Input_Data_sub_iteration,Input_Data_sub[choices]))
    Input_Data_sub = np.delete(Input_Data_sub,choices, axis=0)

    Output_Data_UTM_sub_iteration = np.vstack((Output_Data_UTM_sub_iteration,Output_Data_UTM_sub[choices]))
    Output_Data_UTM_sub = np.delete(Output_Data_UTM_sub,choices, axis=0)
'''




hiddenLayerDimension = 10  ##hidden layer dimension
model = Sequential()
model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=Input_Data_sub.shape[1]))  ##input layer
for i in range(3):
    model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=hiddenLayerDimension))##hidden layers

model.add(Dense(units=hiddenLayerDimension, activation='relu', kernel_regularizer=regularizers.l2(3),input_dim=hiddenLayerDimension)) ##regularize last hidden layer

model.add(Dense(units=Output_Data_UTM_sub.shape[1], activation='linear',  kernel_initializer=keras.initializers.Zeros(),input_dim=hiddenLayerDimension))##output layer
adam=optimizers.Adam(lr=0.0006  , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mse', optimizer=adam, metrics=['mse'])
myWeights = model.get_weights()
for i in range(500):
    #print(myWeights)
    if i != 0:
        model.load_weights('mymodelweights.h5')
    history = model.fit(Input_Data_sub , Output_Data_UTM_sub, validation_split=0.0,epochs=1,batch_size =256, verbose = 2)  ##train and validation is split over 90 % subset

    model.save_weights('mymodelweights.h5')

    #print(myWeights)
    preds = model.predict(Input_Data_Test)  ##predict on initial 10% test subset
    diffs = preds - Output_Data_UTM_Test  ## compute difference between predicted and output values

    avgstraightlinedistpersample = np.sqrt(np.sum(diffs[:,:]**2, axis = 1))  ##returns a vector of values per sample (straight line distance)
    avgStraightLineDist = np.mean(np.sqrt(np.sum(diffs[:,:]**2, axis = 1)))  ##returns a scalar average straight line distance
    print("Samples of training data for this iteration: ",Input_Data_sub.shape)
    print("Iteration Number Average straight line distance is: ", avgStraightLineDist)
    out.append(avgStraightLineDist)

#out = np.append(Output_Data_UTM_Test,np.transpose(np.array([avgstraightlinedistpersample])),axis=1)  ## if repeating experiment
#outToFile = np.array(out)
#np.savetxt("Trial2_TenPerc_Data_Cam2.csv",out,delimiter=',')  ##saving to a file if necessary
np.savetxt("Trial2errorforheatmapAllDataCam2.csv",[Output_Data_UTM_Test[:,0],Output_Data_UTM_Test[:,1],avgstraightlinedistpersample[:]],delimiter=',')
#np.savetxt("output.csv",predictionList,delimiter=',')
