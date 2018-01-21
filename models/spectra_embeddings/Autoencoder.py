# Autoencoder implementations to reduce stellar spectra dimension
from models.spectra_embeddings.Embedding import Embedding
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Conv1D, MaxPooling1D, Flatten, AveragePooling1D

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
		layer_1 = Dense(1000, activation='relu')(input_spectra)
		encoded = Dense(E_D, activation='relu')(layer_1)

		# Decoding
		layer_3 = Dense(1000, activation='relu')(encoded)
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
		self.model.fit(X, X, epochs = 50, batch_size=32, shuffle=True, validation_split=0.1)


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

		# Construct the auto encoder
		input_spectra = Input(shape=(S_D,1), name='input_spectra')
		conv_layer_1 = Conv1D(filters = 8, kernel_size=1, strides=2, activation='relu')(input_spectra)
		conv_layer_1 = MaxPooling1D(pool_size = 2)(conv_layer_1)
		conv_layer_2 = Dropout(0.2)(Conv1D(filters = 8, kernel_size=1, strides=2, activation = 'relu')(conv_layer_1))
		conv_layer_2 = MaxPooling1D(pool_size = 2)(conv_layer_2)
		conv_layer_3 = Dropout(0.2)(Conv1D(filters = 16, kernel_size=2, strides=2, activation = 'relu')(conv_layer_2))
		conv_layer_3 = MaxPooling1D(pool_size = 2)(conv_layer_3)
		conv_layer_4 = Dropout(0.2)(Conv1D(filters = 16, kernel_size=2, strides=2, activation = 'relu')(conv_layer_3))
		conv_layer_4 = MaxPooling1D(pool_size = 8)(conv_layer_4)

		encoded = Flatten(name="encoded")(conv_layer_4)

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
