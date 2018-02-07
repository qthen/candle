# Autoencoder implementations to reduce stellar spectra dimension
from models.spectra_embeddings.Embedding import Embedding
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from keras.optimizers import SGD, Adam

'''
Spectra embedding via simple fully connected auto encoder
'''
class FullyConnectedAutoencoder(Embedding):

	'''
	Constructs a fully connected auto encoder
	S_D - The spectra original dimension
	E_D - The embedding dimension
	'''
	def __init__(self, S_D, E_D = 10):
		self.E_D = E_D

		# Construct the auto encoder
		input_spectra = Input(shape=(S_D,), name='input_spectra')
		encoded = Dense(100, activation='relu')(input_spectra)

		# Decoding
		decoded = Dense(S_D, activation='linear')(encoded)

		self.model = Model(inputs=[input_spectra], output=decoded)
		self.encoding_model = Model(inputs=[input_spectra], output=encoded)

		self.model.summary()

	'''
	Overriding from the Embedding class
	Inputs:
		X - The input spectra
	'''
	def fit(self, X):
		optimizer = SGD(1, momentum=0.9, decay=0.0, nesterov=True)
		self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
		self.model.fit(X, X, epochs = 1000, batch_size=32, shuffle=True, validation_split=0.1)


	'''
	Overriding from the Embedding class
	'''
	def embed(self, X): 
		return self.encoding_model.predict(X)


'''
Spectra embedding via simple convolutional autoencoder
'''
class ConvolutionalAutoencoder(Embedding):

	'''
	Constructs a fully connected auto encoder
	S_D - The spectra original dimension
	E_D - The embedding dimension
	'''
	def __init__(self, S_D, E_D = 10):
		self.E_D = E_D

		# Create the model
		input_spectra = Input(shape=(S_D, 1), name='input_spectra')

		# Dense layers
		conv_layer_1 = Conv1D(filters = 4, kernel_size=8, strides=1, activation='relu')(input_spectra)
		conv_layer_2 = Conv1D(filters = 16, kernel_size=8, strides=1, activation = 'relu')(conv_layer_1)
		conv_layer_2 = MaxPooling1D(pool_size = 4)(conv_layer_2)
		conv_layer_3 = Dropout(0.3)(Conv1D(filters = 16, kernel_size=2, strides=1, activation = 'relu')(conv_layer_2))
		conv_layer_3 = MaxPooling1D(pool_size = 2)(conv_layer_3)
		conv_layer_4 = Dropout(0.3)(Conv1D(filters = 32, kernel_size=2, strides=2, activation = 'relu')(conv_layer_3))
		conv_layer_4 = MaxPooling1D(pool_size = 2)(conv_layer_4)
		conv_layer_5 = Dropout(0.3)(Conv1D(filters = 32, kernel_size=2, strides=2, activation = 'relu')(conv_layer_4))
		conv_layer_5 = MaxPooling1D(pool_size = 2)(conv_layer_5)
		conv_layer_6 = Dropout(0.3)(Conv1D(filters = 32, kernel_size=2, strides=2, activation = 'relu')(conv_layer_5))
		conv_layer_6 = MaxPooling1D(pool_size = 2)(conv_layer_6)
		conv_layer_7 = Dropout(0.3)(Conv1D(filters = 64, kernel_size=2, strides=2, activation = 'relu')(conv_layer_6))
		conv_layer_7 = MaxPooling1D(pool_size = 2)(conv_layer_7)

		encoded = Flatten(name="encoded")(conv_layer_7)

		# Decoding
		layer_3 = Dense(100, activation='relu')(encoded)
		decoded = Dense(S_D, activation='relu')(layer_3)

		self.model = Model(inputs=[input_spectra], output=decoded)
		self.encoding_model = Model(inputs=[input_spectra], output=encoded)

		self.model.summary()

	'''
	Overriding from the Embedding class
	Inputs:
		X - The input spectra
	'''
	def fit(self, X):
		self.model.compile(optimizer='adadelta', loss='mse')
		self.model.fit(X, X.reshape((X.shape[0], X.shape[1])), epochs = 50, batch_size=32, shuffle=True, validation_split=0.1)


	'''
	Overriding from the Embedding class
	'''
	def embed(self, X): 
		return self.encoding_model.predict(X)
