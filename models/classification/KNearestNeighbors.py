from models.Model import Model
from sklearn.neighbors import KNeighborsClassifier
from services.ModelPerformanceVisualization import plot_classification
import numpy as np

class KNearestNeighbors(Model):

	'''
	Init a simple Gaussian Naive Bayes model
	'''
	def __init__(self):
		self.model = KNeighborsClassifier()


	'''
	Compiles the model
	'''
	def compile(self):
		pass

	'''
	Fits the gaussian naive bayes model to X
	Input:
		X - Stellar spectra
	Throws:
		TypeError - If model is fit before it has been compiled
	'''
	def fit(self, X, y):
		self.model.fit(X, y)

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
	Scores the model given the stellar spectra and labels and prints it out
	Input:
		X - Stellar spectra
		y - Labels: 1|0
	'''
	def score(self, X, y):
		mean_acc = self.model.score(X, y)
		print("K-Nearest Neighbors binary classification MAE: {}".format(mean_acc))

	'''
	Evaluates and judges itself then plots visualizations of how well it did on predictions on the given data and the target values
	Input:
		X - The spectra from Kepler
		y - The target PS and Δv as a list and ground truth
	'''
	def judge(self, X, y):
		plot_classification(y[1], y[0], [int(i) for i in self.predict(X)], class_labels=['Red giant branch', 'Red clumps'], title="K-Nearest Neighbors classification")


