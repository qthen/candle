from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Conv1D, MaxPooling1D, Flatten

'''
A simple baseline convolutional network that has two main steps:
	1. Learn embedding of star spectra to bottleneck layers
	2. Using the embedding, preform regression on asterosesmic parameters (separately for now)
'''
class BaselineCNN(object):

	'''
	Creates the baseline CNN with separate decodings (step 2)
	Inputs:
		S_D - Dimension of the spectra data
	'''
	def __init__(self, S_D = 10):
		# Create the model
		input_spectra = Input(shape=(S_D, 1), name='input_spectra')

		# Dense layers
		conv_layer_1 = Conv1D(filters = 15, kernel_size=3, activation='relu')(input_spectra)
		conv_layer_1 = MaxPooling1D(pool_size = 8)(conv_layer_1)
		conv_layer_2 = Dropout(0.2)(Conv1D(filters = 32, kernel_size=3, activation = 'relu')(conv_layer_1))
		conv_layer_2 = MaxPooling1D(pool_size = 8)(conv_layer_2)
		conv_layer_3 = Dropout(0.2)(Conv1D(filters = 64, kernel_size=3, activation = 'relu')(conv_layer_2))
		conv_layer_3 = MaxPooling1D(pool_size = 8)(conv_layer_3)
		conv_layer_4 = Conv1D(filters = 50, kernel_size=3, activation = 'relu')(conv_layer_3)
		conv_layer_4 = MaxPooling1D(pool_size = 2)(conv_layer_4)

		embedding = Flatten(name='embedding')(conv_layer_4)

		dense_layer_1 = Dense(64, activation='relu')(embedding)
		dense_layer_2 = Dense(32, activation='relu')(dense_layer_1)
		prediction_dnu = Dense(1, activation='relu', name='Dnu')(dense_layer_2)

		dense_layer_1 = Dense(64, activation='relu')(embedding)
		dense_layer_2 = Dense(32, activation='relu')(dense_layer_1)
		prediction_ps = Dense(1, activation='relu', name='DPi1')(dense_layer_2)

		self.model = Model(inputs=[input_spectra], output=[prediction_ps, prediction_dnu])

		self.model.summary()

	'''
	Compile the model given the optimizer, loss function, and the model architecture and metrics
	Input:
		loss_fn - The loss function to use
		optimizer - The optimizer to use
		metrics - Metrics for the model
	Returns:
		None
	'''
	def compile(self, loss_fn='mse', optimizer = 'adam', metrics = ['mae', 'mape']):
		self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)


	'''
	Fits the model given the input data X and asteroseismic features y (DPi1, Dnu) and returns the history of the model
	Input:
		X - Stellar spectra (m x d) ndarray
		y - Asteroseismic parameters (m x 2) ndarray
		epochs - The number of epochs for this model
		batch_size - Batch size to train
		validation_split - Validation split percentage
	Returns:
		History object
	'''
	def fit(self, X, y, epochs = 10, batch_size = 32, validation_split = 0.1):
		return self.model.fit(X, y, epochs=epochs, batch_size = batch_size, validation_split = validation_split, shuffle = True)