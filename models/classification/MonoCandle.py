'''
A vectorized implementation of a one layer hidden neural network on the entire spectra without any compression
'''
import autograd.numpy as np
from autograd import grad
import pickle

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-1*z))

def relu(z):
	return np.maximum(z, 0)


'''
The model can be described as follows:
y_pred = σ(ReLU(XW_1 + b_1)W_2 + b_2)
'''
class MonoCandle(object):

	PICKLE_FP = "models/classification/MonoCandle.pickle"

	'''
	Initializes the model with set weights from the pickle file
	'''
	def __init__(self):
		pickle_out = open(MonoCandle.PICKLE_FP, 'rb')
		self.params = pickle.load(pickle_out)
		self.W_1, self.b_1 = self.params[0]
		self.W_2, self.b_2 = self.params[1]


	def _predict(self, weights, X):
		W_1, b_1 = weights[0]
		W_2, b_2 = weights[1]
		return sigmoid(np.dot(relu(np.dot(X, W_1) + b_1), W_2) + b_2)

	'''
	Given a dataset matrix X, returns predictions, flattened in size (N, )
	Input:
		X - Data matrix ∈ (N, 7514)
	Outputs:
		y - Vector ∈ (N, )
	'''
	def predict(self, X):
		return self._predict((self.W_1, self.b_1), (self.W_2, self.b_2), X)

	'''
	Returns grad function for gradients of weights w.r.t to some input
	'''
	def input_grad():
		return grad(self.predict)

	'''
	Returns grad function for gradients of some input w.r.t to current weights
	'''
	def weight_grad(X):
		return grad(lambda weights: self._predict(weights, X))




