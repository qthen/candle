from models.Model import Model
from sklearn.mixture import GaussianMixture
from services.ModelPerformanceVisualization import plot_classification
import matplotlib.pyplot as plt
import numpy as np

class MixtureOfGaussian(Model):

	'''
	Init a Gaussian Mixture model
	Inputs:
		n_components = Number of mixture components
	'''
	def __init__(self, n_components = 3):
		self._K = n_components
		self.model = None


	'''
	Compiles the model
	Inputs:
		max_iterations - The maximum iterations to train for
	'''
	def compile(self, max_iterations = 25):
		self.model = GaussianMixture(n_components = self._K, max_iter = max_iterations)

	'''
	Fits the mixture model to X
	Input:
		X - Stellar spectra
	Throws:
		TypeError - If model is fit before it has been compiled
	'''
	def fit(self, X):
		if self.model:
			return self.model.fit(X)
		else:
			raise TypeError("Model not compiled yet")

	'''
	Predicts on X and returns the class labels
	Input:
		X - Stellar spectra
	Throws:
		TypeError - If model is predicted before it has been compiled
	'''
	def predict(self, X):
		if self.model:
			return self.model.predict(X)
		else:
			raise TypeError("Model not compiled yet")

	'''
	Evaluates and judges itself then plots visualizations of how well it did on predictions on the given data and the target values
	Input:
		X - The spectra from Kepler
		y - The target PS and Δv as a list and ground truth
	'''
	def judge(self, X, y):
		predictions = [int(i) for i in self.predict(X)]
		title = "Clustering of Kepler stars with {} mixtures".format(self._K)
		labels = ["Class: {}".format(i) for i in range(0, self._K)]
		plot_classification(y[1], y[0], predictions, class_labels=labels, title=title)

