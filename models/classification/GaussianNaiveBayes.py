from models.Model import Model
from sklearn.naive_bayes import GaussianNB
from services.ModelPerformanceVisualization import plot_classification
import numpy as np

class GaussianNaiveBayes(Model):

	'''
	Init a simple Gaussian Naive Bayes model
	'''
	def __init__(self):
		self.model = GaussianNB()


	'''
	Compiles the model
	'''
	def compile(self, max_iterations = 100):
		pass

	'''
	Fits the gaussian naive bayes model to X
	Input:
		X - Stellar spectra
	Throws:
		TypeError - If model is fit before it has been compiled
	'''
	def fit(self, X, y):
		return self.model.fit(X, y)

	'''
	Predicts on X and returns the class labels
	Input:
		X - Stellar spectra
	Throws:
		TypeError - If model is predicted before it has been compiled
	'''
	def predict(self, X):
		return self.model.predict(X)

	'''
	Evaluates and judges itself then plots visualizations of how well it did on predictions on the given data and the target values
	Input:
		X - The spectra from Kepler
		y - The target PS and Δv as a list and ground truth
	'''
	def judge(self, X, y):
		plot_classification(y[1], y[0], [int(i) for i in self.predict(X)], class_labels=['Red giant branch', 'Red clumps'])


