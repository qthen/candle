from models.Model import Model
from sklearn.linear_model import LogisticRegression as LR
from services.ModelPerformanceVisualization import plot_classification
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression(Model):

	'''
	Init a simple Logistic Regression model
	'''
	def __init__(self):
		self.model = None


	'''
	Compiles the model
	Inputs:
		max_iterations - The maximum iterations to train for
	'''
	def compile(self, max_iterations = 100):
		self.model =  LR(max_iter = max_iterations)

	'''
	Fits the mixture model to X
	Input:
		X - Stellar spectra
	Throws:
		TypeError - If model is fit before it has been compiled
	'''
	def fit(self, X, y):
		if self.model:
			return self.model.fit(X, y)
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
		plot_classification(y[1], y[0], [int(i) for i in self.predict(X)], class_labels=['Red giant branch', 'Red clumps'])


