import cv2
import os
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,AveragePooling2D
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import time
from keras.layers.core import Lambda
from keras.engine.topology import Layer
import tensorflow as tf
import sys
sys.path.append("J:\神经网络/tensorflow")
import input_data
from keras.models import Model
from keras import backend as K

class channel_average_pooling(Layer):
	def __init__(self,output_dim,**km):
		self.output_dim = output_dim
		self.scaler = None
		super(channel_average_pooling,self).__init__(**km)
	def build(self,input_shape):
		shape =list(input_shape)
		k = shape[-1]
		self.scaler=self.add_weight(name = 'scaler',shape = (k,),initializer = 'uniform',trainable = True)
		super(channel_average_pooling,self).build([k])
	def call(self,img):
		y=tf.reduce_sum(img*self.scaler,axis =3)
		print(y)
		return y
	def compute_output_shape(self,input_shape):
		shape = list(input_shape)
		shape[-1]=1
		print(shape)
		return tuple(shape)


def kernel(shape):
	kernel = K.ones(shape)*0.01
	return kernel


class cnn_features:
	def __init__(self,batch_size =5,epoch = 10,classes=10):
		self.batch_size = batch_size
		self.epoch = epoch
		self.classes = classes

	def train(self,train,label):
		input_shape = (train.shape[1],train.shape[2],train.shape[3])
		self.model = Sequential()
		self.model.add(Conv2D(32,(3,3),activation = "relu",input_shape= input_shape,padding = 'SAME'))
		self.model.add(MaxPooling2D((2,2)))
		self.model.add(Dropout(0.2))

		self.model.add(Conv2D(16,(3,3),activation = 'relu',padding = 'SAME'))
		self.model.add(MaxPooling2D((2,2)))
		self.model.add(Dropout(0.2))

		input_shape =self.model.get_output_shape_at(0)
		output_shape =list(input_shape)
		output_shape[-1]=1
		print(output_shape,input_shape)
		self.model.add(channel_average_pooling(output_shape,input_shape =input_shape))
		#model.add(Lambda(channel_average_pooling))
		print("#####",self.model.get_output_shape_at(0))

		self.model.add(Flatten())
		self.model.add(Dense(128,activation = 'relu',name = 'features'))
		self.model.add(Dropout(0.2))
		# model.add(Dense(512,activation = 'relu'))
		# model.add(Dropout(0.2))

		self.model.add(Dense(self.classes,activation = 'softmax'))
		self.model.compile(optimizer = keras.optimizers.SGD(lr = 0.01),
					 loss = keras.losses.categorical_crossentropy,
					 metrics = ['accuracy'])
		history = self.model.fit(train,label,batch_size = self.batch_size,epochs = self.epoch)
	def predict(self,test):
		pre = self.model.predict(test)
		return pre
	def features(self,data):
		model = Model(input = self.model.input,output = self.model.get_layer('features').output)
		feature = model.predict(data)
		return feature




if __name__ == "__main__":
	start = time.time()
	minist=input_data.read_data_sets("MNIST_data/",one_hot=True)
	train = minist.train.images
	train = train.reshape((-1,28,28,1))
	train_label = minist.train.labels

	test = minist.test.images
	test = test.reshape((-1,28,28,1))
	test_label =minist.test.labels

	model = cnn_features()
	model.train(train,train_label)
	pre =model.predict(test)
	print(np.cast['float'](np.argmax(pre,axis =1)==np.argmax(test_label,axis =1)).mean())
	print(time.time()-start)

