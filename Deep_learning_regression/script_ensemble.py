#imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time 
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, SimpleRNN
import tensorflow as tf
import csv
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import glob


def transformPCA(combined_data, cols_to_norm, components):
	pca = PCA(n_components=components)
	# prepare transform on dataset
	pca.fit(combined_data[cols_to_norm])
	# apply transform to dataset
	transformed = pca.transform(combined_data[cols_to_norm])
	return pd.DataFrame(transformed)

def transformSVD(combined_data, cols_to_norm, components):
	svd = TruncatedSVD(n_components=components)
	# prepare transform on dataset
	svd.fit(combined_data[cols_to_norm])
	# apply transform to dataset
	transformed = svd.transform(combined_data[cols_to_norm])
	return pd.DataFrame(transformed)

def model11(components):
	# Designing and initializing the model.
	model1 = Sequential()
	model1.add(SimpleRNN(100, input_shape = (1,components), dropout = 0.2, return_sequences=True))
	model1.add(SimpleRNN(80, dropout = 0.2, return_sequences=True))
	model1.add(SimpleRNN(60, dropout = 0.2, return_sequences=False))
	model1.add(Dense(1, activation = 'relu'))
	model1.compile(loss = 'mse' , optimizer = 'adam' , metrics = ['mse', 'mae', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredLogarithmicError()] )

	return model1

def model31(components):
	# Designing and initializing the model.
	model = Sequential()
	model.add(LSTM(100, input_shape = (1,components), dropout = 0.2, return_sequences=True))
	model.add(LSTM(80, dropout = 0.2, return_sequences=True))
	model.add(LSTM(60, dropout = 0.2, return_sequences=True))
	model.add(LSTM(40, dropout = 0.2, return_sequences=True))
	model.add(LSTM(20, dropout = 0.2, return_sequences=False))
	model.add(Dense(1, activation = 'relu'))
	model.compile(loss = 'mse' , optimizer = 'adam' , metrics = ['mse', 'mae', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredLogarithmicError()] )

	return model

def model24(components):
	#Building the model
	model = Sequential()

	#add model layers
	model.add(Conv2D(64, kernel_size=1, activation='relu',input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=1, activation='relu'))
	model.add(Conv2D(16, kernel_size=1, activation='relu'))
	    
	    
	#model.add(MaxPool2D(pool_size=(1,8)))
	#model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='relu'))

	#compile model using mse as the loss function
	model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredLogarithmicError()])

	return model

def model25(components):
	# Designing and initializing the model.
	model = Sequential()
	model.add(GRU(100, input_shape = (1,components), dropout = 0.2, return_sequences=True))
	model.add(GRU(80, dropout = 0.2, return_sequences=True))
	model.add(GRU(60, dropout = 0.2, return_sequences=False))
	model.add(Dense(1, activation = 'relu'))
	model.compile(loss = 'mse' , optimizer = 'adam' , metrics = ['mse', 'mae', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredLogarithmicError()] )

	return model

def model(combined_data, cols_to_norm, train_data_index_list, test_data_index_list, components, transformed, model):
	X_train1 = transformed[transformed.index.isin(train_data_index_list)]
	X_train1 = np.array(X_train1)

	X_test1 = transformed[transformed.index.isin(test_data_index_list)]
	X_test1 = np.array(X_test1)

	Y_train1 = combined_data[transformed.index.isin(train_data_index_list)]
	Y_train1 = Y_train1['bug']

	Y_test1 = combined_data[transformed.index.isin(test_data_index_list)]
	Y_test1 = Y_test1['bug']

	# Applying over sampling and SMOTE to training data for augmentation.
	ros = RandomOverSampler(random_state=0)
	X_train1, Y_train1 = ros.fit_resample(X_train1, Y_train1)

	smt = SMOTE()
	X_train1, Y_train1 = smt.fit_resample(X_train1, Y_train1)

	train_x1 = np.reshape(X_train1, (X_train1.shape[0], 1, X_train1.shape[1]))
	test_x1 = np.reshape(X_test1, (X_test1.shape[0], 1, X_test1.shape[1]))

	train_y1 = Y_train1.to_numpy()
	test_y1 = Y_test1.to_numpy()

	# Fitting the model on training data.
	history = model.fit(train_x1, train_y1, epochs = 100, batch_size = 128)

	return model.predict(test_x1)


if __name__ == "__main__":
	files = glob.glob("../../datasets/ant-*.csv", recursive = True)
	combined_data = pd.concat(map(pd.read_csv, files))

	# Applying Min Max Scaling.
	scaler = MinMaxScaler()
	MinMaxScaler(copy=True, feature_range=(0, 1))
	cols_to_norm = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']
	combined_data[cols_to_norm] = MinMaxScaler().fit_transform(combined_data[cols_to_norm])

	test_length = len(combined_data[(combined_data['version']==1.7)])
	test_data_index_list = list(range(test_length))

	total_length = len(combined_data)
	train_data_index_list = list(range(test_length, test_length + total_length))
	
	components1 = 10
	components2 = 15

	transformed1 = transformPCA(combined_data, cols_to_norm, components1)
	transformed2 = transformPCA(combined_data, cols_to_norm, components2)
	transformed3 = transformSVD(combined_data, cols_to_norm, components1)

	model1 = model11(components1)
	model2 = model31(components2)
	model3 = model24(components1)
	model4 = model11(components1)
	model5 = model25(components1)

	predictions_y1 = np.rint(model(combined_data, cols_to_norm, train_data_index_list, test_data_index_list, components1, transformed1, model1))
	predictions_y2 = np.rint(model(combined_data, cols_to_norm, train_data_index_list, test_data_index_list, components2, transformed2, model2))
	predictions_y3 = np.rint(model(combined_data, cols_to_norm, train_data_index_list, test_data_index_list, components1, transformed1, model3))
	predictions_y4 = np.rint(model(combined_data, cols_to_norm, train_data_index_list, test_data_index_list, components1, transformed3, model4))
	predictions_y5 = np.rint(model(combined_data, cols_to_norm, train_data_index_list, test_data_index_list, components1, transformed1, model5))

	print("success!!")

