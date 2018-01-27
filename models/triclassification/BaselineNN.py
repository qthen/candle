from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Model, load_model

'''
A baseline neural network that consists of separate fully connected layers to a given star spectra
Model meant for triclassification (RGB, RC, Secondary clumps)

Star spectra for this model should be dimensionally reduced prior to prediction due to the size of a large 
fully connected layer and low count of training data
'''
class BaselineNN(object):

	# Model filepath for saving models
	MODEL_FILEPATH = "models/triclassification/BaselineNN.h5"

	'''
	Creates the baseline NN with separate fully connected layers on stellar spectra
	Inputs:
		S_D - Dimension of spectra data
	'''
	def __init__(self, S_D = 10):
		self.S_D = S_D
		
		# Create the model
		input_spectra = Input(shape=(S_D,), name='input_spectra')

		# Dense layers for PS - not shared for now
		dense_layer_1 = Dropout(0.1)(Dense(128, activation='elu')(input_spectra))
		dense_layer_2 = Dropout(0.1)(Dense(64, activation='elu')(dense_layer_1))
		prediction = Dense(3, activation='softmax', name='prediction')(dense_layer_2)

		self.model = Model(inputs=[input_spectra], output=prediction)

	'''
	Compile the model given the optimizer, loss function, and the model architecture and metrics
	Input:
		loss_fn - The loss function to use
		optimizer - The optimizer to use
		metrics - Metrics for the model
	Returns:
		None
	'''
	def compile(self, loss_fn='categorical_crossentropy', optimizer = 'adam', metrics = ['acc']):
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


	'''
	Predicts on a given test data sample
	Inputs:
		X - The spectra data
	Returns:
		array of nparray, [0] => PS, [1] => Δv
	'''
	def predict(self, X):
		return self.model.predict(X)

	'''
	Saves the model into its constant defined file
	'''
	def save(self):
		self.model.save(BaselineNN.MODEL_FILEPATH)

	'''
	If called, loads the current saved model as the starting point
	'''
	def load(self):
		self.model = load_model(BaselineNN.MODEL_FILEPATH)