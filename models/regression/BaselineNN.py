from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Model
import services.ModelPerformanceVisualization as model_visualizer

'''
A baseline neural network that consists of separate fully connected layers to a given star spectra and 
does regression on asteroseismic parameters

Star spectra for this model should be dimensionally reduced prior to prediction due to the size of a large 
fully connected layer and low count of training data

100 epoch result
loss                                    : 1611.0846
DPi1_loss                               : 1610.5841
Dnu_loss                                : 0.5005
DPi1_mean_absolute_error                : 28.6636
DPi1_mean_absolute_percentage_error     : 17.2518
Dnu_mean_absolute_error                 : 0.4684
Dnu_mean_absolute_percentage_error      : 7.1293
val_loss                                : 1818.0663
val_DPi1_loss                           : 1817.6984
val_Dnu_loss                            : 0.3680
val_DPi1_mean_absolute_error            : 27.0088
val_DPi1_mean_absolute_percentage_error : 14.0153
val_Dnu_mean_absolute_error             : 0.3383
val_Dnu_mean_absolute_percentage_error  : 5.6662
'''
class BaselineNN(object):

	'''
	Creates the baseline NN with separate fully connected layers on stellar spectra
	Inputs:
		S_D - Dimension of spectra data
	'''
	def __init__(self, S_D = 10):
		# Create the model
		input_spectra = Input(shape=(S_D,), name='input_spectra')

		# Dense layers for PS - not shared for now
		dense_layer_1 = Dropout(0.1)(Dense(128, activation='relu')(input_spectra))
		dense_layer_2 = Dropout(0.1)(Dense(64, activation='relu')(dense_layer_1))
		prediction_ps = Dense(1, activation='relu', name='DPi1')(dense_layer_2)

		# Dense layers for Dnu - not shared for now
		dense_layer_1 = Dropout(0.1)(Dense(128, activation='relu')(input_spectra))
		dense_layer_2 = Dropout(0.1)(Dense(64, activation='relu')(dense_layer_1))
		prediction_dnu = Dense(1, activation='linear', name='Dnu')(dense_layer_2)

		self.model = Model(inputs=[input_spectra], outputs=[prediction_ps, prediction_dnu])

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
	Evaluates and judges itself then plots visualizations of how well it did on predictions on the given data and the target values
	Input:
		X - The spectra data
		y - The target PS and Δv as a list
	Returns:
		Boolean - True on success
	'''
	def judge(self, X, y):
		y_PS, y_Dnu = y
		y_pred_PS, y_pred_Dnu = self.predict(X)
		model_visualizer.plot_ps_vs_pred_ps(y_PS, y_pred_PS)
		model_visualizer.plot_dnu_vs_pred_dnu(y_Dnu, y_pred_Dnu)



