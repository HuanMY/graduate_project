import keras
from keras.layers import *
from keras.models import *
from split_video import *
from sklearn.model_selection import KFold
from keras.callbacks import LearningRateScheduler,Callback

def Conv(conv1):
	conv1 = Conv2D(32,(3,3),strides = (1,1),padding = 'same',activation = 'relu')(conv1)
	maxpool1 = MaxPooling2D()(conv1)
	conv1 = Conv2D(64,(3,3),strides = (1,1),padding = 'same',activation = 'relu')(maxpool1)
	conv1 = Conv2D(64,(3,3),strides = (1,1),padding = 'same',activation = 'relu')(conv1)
	maxpool1 = MaxPooling2D()(conv1)
	conv1 = Conv2D(256,(3,3),strides = (1,1),padding = 'same',activation = 'relu')(maxpool1)
	maxpool1 = MaxPooling2D()(conv1)
	glob = GlobalAveragePooling2D()(maxpool1)
	return glob
def bilstm_model():
	inputs = Input(shape = train[0].shape)
	emb = concatenate([Lambda(lambda x:K.expand_dims(x,1))(Conv(Lambda(lambda x:x[:,i,:,:,:])(inputs))) for i in range(n)],axis = 1)

	x = SpatialDropout1D(0.2)(emb)
	bilstm = Bidirectional(LSTM(256, return_sequences=True,name = 'lstm1'),merge_mode = 'concat')(x)
	bilstm = Bidirectional(LSTM(256, return_sequences=True,name = 'lstm2'),merge_mode = 'concat')(bilstm)

	hidden = concatenate([
		GlobalMaxPooling1D()(bilstm),
		GlobalAveragePooling1D()(bilstm),
		])
	hidden = add([hidden, Dense(1024, activation='relu')(hidden)])
	hidden = add([hidden, Dense(1024, activation='relu')(hidden)])

	dense = Dense(1024,activation = 'relu')(hidden)
	dense = concatenate([hidden,dense])
	outputs = Dense(3,activation = 'softmax')(dense)
	model = Model(inputs,outputs)
	model.compile(loss = keras.losses.binary_crossentropy,
				  optimizer = keras.optimizers.Adam(1e-3),
				  metrics = ['accuracy'])
	for epoch in range(epochs):
		history = model.fit(train,train_label,
				 epochs = 1,
				 batch_size = 10,
				 validation_data = (val,val_label),
	            callbacks=[
	            LearningRateScheduler(lambda x: 1e-3 * (0.55 ** epoch) if 1e-3 * (0.55 ** epoch)>1e-5 else 1e-5)
	            ]
			 	)
	return model

if __name__=='__main__':
	epochs = 5
	train,label = get_data()
	train = train/255
	kfold = KFold(5,True,2019)
	train_ind,val_ind = next(kfold.split(train))
	train,label = np.array(train),np.array(label)
	label = np.eye(3)[label]
	train,val = train[train_ind],train[val_ind]
	train_label,val_label = label[train_ind],label[val_ind]
	n = len(train[0])
	model = bilstm_model()





