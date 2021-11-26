import tflearn
import tensorflow.compat.v1 as tf

def get_model(IMG_SIZE,no_of_fruits,LR):
	try:
		tf.reset_default_graph()
	except:
		print("tensorflow")
	convnet = tflearn.layers.core.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

	convnet = tflearn.layers.conv.conv_2d(convnet, 32, 5, activation='relu')

	convnet = tflearn.layers.conv.max_pool_2d(convnet, 5)

	convnet = tflearn.layers.conv.conv_2d(convnet, 64, 5, activation='relu')

	convnet = tflearn.layers.conv.max_pool_2d(convnet, 5)

	convnet = tflearn.layers.conv.conv_2d(convnet, 128, 5, activation='relu')
	convnet = tflearn.layers.conv.max_pool_2d(convnet, 5)

	convnet = tflearn.layers.conv.conv_2d(convnet, 64, 5, activation='relu')
	convnet = tflearn.layers.conv.max_pool_2d(convnet, 5)

	convnet = tflearn.layers.conv.conv_2d(convnet, 32, 5, activation='relu')
	convnet = tflearn.layers.conv.max_pool_2d(convnet, 5)

	convnet = tflearn.layers.core.fully_connected(convnet, 1024, activation='relu')
	convnet = tflearn.layers.core.dropout(convnet, 0.8)

	convnet = tflearn.layers.core.fully_connected(convnet, no_of_fruits, activation='softmax')
	convnet = tflearn.layers.estimator.regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.models.dnn.DNN(convnet, tensorboard_dir='log')

	return model

def get_model_keras(IMG_SIZE,no_of_fruits,LR):
	# Importing the Keras libraries and packages
	from tensorflow import keras
	from keras.layers import Conv2D
	from keras.layers import MaxPooling2D
	from keras.layers import Flatten
	from keras.layers import Dense, Dropout
	from keras.models import Sequential

	# Initialising the CNN
	classifier = Sequential()

	classifier.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))
	classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))

	classifier.add(Conv2D(64, (3, 3), activation='relu' ))
	classifier.add(Conv2D(64, (3, 3), activation='relu'))
	classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	classifier.add(Flatten())

	classifier.add(Dense(units = 512, activation = 'relu')) 
	classifier.add(Dropout(0.4))
	classifier.add(Dense(units = 256, activation = 'relu')) 
	classifier.add(Dropout(0.4))
	classifier.add(Dense(units = no_of_fruits, activation = 'softmax'))

	opt = keras.optimizers.Adam(learning_rate=LR)
	classifier.compile(optimizer = opt, loss='categorical_crossentropy', metrics = ['accuracy'])

	return classifier

def get_model_keras_vgg16(IMG_SIZE,no_of_fruits,LR):
	# Importing the Keras libraries and packages
	from tensorflow import keras
	from keras.layers import Conv2D
	from keras.layers import MaxPooling2D
	from keras.layers import Flatten
	from keras.layers import Dense ,Dropout
	from keras.models import Sequential

	model = Sequential()

	# Step 1 - Convolution
	model.add(Conv2D(filters=64, 
              kernel_size=(3, 3), 
              padding='same', 
              activation='relu', 
              input_shape=(224,224,3), 
              name='conv1_1'))
	model.add(Conv2D(filters=64, 
              kernel_size=(3, 3), 
                  padding='same', 
              activation='relu',
              name='conv1_2'))
	model.add(MaxPooling2D(pool_size=(2,2), 
                   strides=(2,2), 
                   name='max_pooling2d_1'))

	model.add(Conv2D(filters=128, 
              kernel_size=(3, 3), 
              padding='same', 
              activation='relu', 
              name='conv2_1'))
	model.add(Conv2D(filters=128, 
              kernel_size=(3, 3), 
                  padding='same', 
              activation='relu',
              name='conv2_2'))
	model.add(MaxPooling2D(pool_size=(2,2), 
                 strides=(2,2), 
                 name='max_pooling2d_2'))

	model.add(Conv2D(filters=256, 
              kernel_size=(3, 3), 
              padding='same', 
              activation='relu', 
              input_shape=(224,224,3), 
              name='conv3_1'))
	model.add(Conv2D(filters=256, 
              kernel_size=(3, 3), 
                  padding='same', 
              activation='relu',
              name='conv3_2'))
	model.add(Conv2D(filters=256, 
              kernel_size=(3, 3), 
                  padding='same', 
              activation='relu',
              name='conv3_3'))
	model.add(MaxPooling2D(pool_size=(2,2), 
                   strides=(2,2), 
                   name='max_pooling2d_3'))

	model.add(Conv2D(filters=512, 
              kernel_size=(3, 3), 
              padding='same', 
              activation='relu', 
              input_shape=(224,224,3), 
              name='conv4_1'))
	model.add(Conv2D(filters=512, 
              kernel_size=(3, 3), 
                  padding='same', 
              activation='relu',
              name='conv4_2'))
	model.add(Conv2D(filters=512, 
              kernel_size=(3, 3), 
                  padding='same', 
              activation='relu',
              name='conv4_3'))
  
	model.add(MaxPooling2D(pool_size=(2,2), 
                   strides=(2,2), 
                   name='max_pooling2d_4'))
	
	model.add(Conv2D(filters=512, 
              kernel_size=(3, 3), 
              padding='same', 
              activation='relu', 
              input_shape=(224,224,3), 
              name='conv5_1'))
	model.add(Conv2D(filters=512, 
              kernel_size=(3, 3), 
                  padding='same', 
              activation='relu',
              name='conv5_2'))
	model.add(Conv2D(filters=512, 
             kernel_size=(3, 3), 
                 padding='same', 
             activation='relu',
             name='conv5_3'))
  
	model.add(MaxPooling2D(pool_size=(2,2), 
                   strides=(2,2), 
                   name='max_pooling2d_5'))
	
	model.add(Flatten(name='flatten'))
  
	model.add(Dense(4096, activation='relu', name='fc_1'))
	model.add(Dropout(0.5, name='dropout_1'))
  
	model.add(Dense(4096, activation='relu', name='fc_2'))
	model.add(Dropout(0.5, name='dropout_4'))
	
	model.add(Dense(1000, activation='softmax', name='output'))

	# Compiling the CNN
	#adam is for stochastic gradient descent 
	opt = keras.optimizers.Adam(learning_rate=LR)
	model.compile(optimizer = opt, loss='categorical_crossentropy', metrics = ['accuracy'])

	return model

if __name__== "__main__":
	get_model_keras(400,3,0.01)