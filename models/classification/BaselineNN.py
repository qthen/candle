from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Model, load_model
from services.ModelPerformanceVisualization import plot_classification
import matplotlib.pyplot as plt

'''
A baseline neural network that consists of separate fully connected layers to a given star spectra and 
does regression on asteroseismic parameters, and is easily adaptable to regression or binary classification

Star spectra for this model should be dimensionally reduced prior to prediction due to the size of a large 
fully connected layer and low count of training data
'''
class BaselineNN(object):

	# Model filepath for saving models
	MODEL_FILEPATH = "models/classification/BaselineNN.h5"

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
		prediction = Dense(1, activation='sigmoid', name='RC')(dense_layer_2)

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
	def compile(self, loss_fn='binary_crossentropy', optimizer = 'adam', metrics = ['acc']):
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
	Scores the model on the given spectra and labels
	Inputs:
		X - Stellar spectra
		y - Labels 1|0
	'''
	def score(self, X, y):
		y_pred = self.predict(X)
		y_pred = [1.0 if prob>= 0.5 else 0.0 for prob in y_pred]
		acc = float(sum([1 if y_pred[i] == y[i] else 0 for i in range(0, len(y))]))/float(len(y))
		print("Baseline neural network binary classification MAE: {}".format(acc))

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
		X - The spectra from Kepler and KIC as a tuple
		y - The target PS and Δv as a list and ground truth
	'''
	def judge(self, X, y):
		plot_classification(y[1], y[0], [1 if i >= 0.5 else 0 for i in self.predict(X)], class_labels=['Red giant branch', 'Red clumps'], title="Baseline Neural Network classification")
		# y_PS, y_Dnu, y_RC = y
		# spectra, kic = X
		# y_rc = self.predict(spectra)
		# RC_COLOR = "#F03434"
		# RGB_COLOR = "#F89406"
		# y_rc_dnu = []
		# y_rgb_dnu = []
		# rcs = []
		# rgbs = []
		# # Get all predicted RC's
		# for i in range(0, len(y_rc)):
		# 	if y_rc[i] >= 0.5:
		# 		rcs.append(y_PS[i])
		# 		y_rc_dnu.append(y_Dnu[i])
		# 	else:
		# 		y_rgb_dnu.append(y_Dnu[i])
		# 		rgbs.append(y_PS[i])
		# ax = plt.subplot(111)
		# rc_plot = plt.scatter(y_rc_dnu, rcs, c="#F03434", alpha = 0.6)
		# rgb_plot = plt.scatter(y_rgb_dnu, rgbs, c="#F89406", alpha = 0.6)
		# plt.plot([0, 20], [100, 175], linestyle='--', color='#013243')
		# plt.xlim(xmin=0, xmax=20)
		# plt.ylim(ymin=0, ymax=400)
		# plt.xlabel("Δv - large frequency separation")
		# plt.ylabel("Period spacing")
		# plt.title("Baseline NN classification on red giants with {} components".format(self.S_D))
		# plt.legend((rc_plot,rgb_plot), ['RC stars', 'RGB stars'])

		# # Annotate misclassified
		# for i in range(0, len(kic)):
		# 	pred = 1 if y_rc[i] >= 0.5 else 0
		# 	if y_RC[i] != pred:
		# 		plt.annotate("{}".format(kic[i]), (y_Dnu[i], y_PS[i]))
		# plt.show()

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