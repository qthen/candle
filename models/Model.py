'''
Base class for a model that all models should inherit from (whether supervised or unsupervised, regression or classification)
'''

class Model(object):

	'''
	Fits the model on given data format X and target y, where the format is agnostic to this method
	Input:
		X - Data, any format
		y - Targets, any format
	'''
	def fit(self, X, y):
		raise NotImplementedError("Abstract method in base class of Model should be overriden")


	'''
	Compile the model, this should be called before fitting and do all preparation
	Inputs:
		Any configurations the model needs (lr, components, etc.)
	'''
	def compile(self):
		raise NotImplementedError("Abstract method in base class of Model should be overriden")

	'''
	With the current model state, predict X
	Input:
		X - Data, any format
		y - Targets, any format
	'''
	def predict(self, X, y):
		raise NotImplementedError("Abstract method in base class of Model should be overriden")

	'''
	Score the results of the model on some validation set
	Input:
		X - Data, any format
		y - Targets, any format
	'''
	def score(self, X, y):
		raise NotImplementedError("Abstract method in base class of Model should be overriden")