from keras.models import load_model
from models.spectra_embeddings.NormalizedPCA import NormalizedSpectralEmbeddingPCA
import numpy as np
import pickle


'''
A final compiled model that is used to classify RC stars from APOGEE red giants

Classifies by regressing stars into Teff-log(g) space, then applying a cut to the theoretical area of red clump stars
Anything outside is considered to be a not RC star (often times, a RGB in the case of APOGEE data sets)

Then runs a 2 hidden layer neural network on the stars inside the cut
'''

class Candle(object):

	# File paths to pretrained models
	APOGEE_MODEL_FP = "models/regression/ApogeeNN.h5"
	IMPROVED_NN_MODEL_FP = "models/classification/ImprovedNN.h5"

	# Constants for the mean and std Teff and logg from ApogeNN
	TEFF_MEAN = 4585.5703125
	TEFF_STD = 383.7852783203125
	LOGG_MEAN = 2.2114694118499756
	LOGG_STD = 0.7215081453323364

	# Constants for the cuts
	MAX_RC_TEFF = 5500.0
	MIN_RC_TEFF = 4500.0
	MAX_RC_LOGG = 3.5
	MIN_RC_LOGG = 2.0

	# Constants for the cuts normalized
	MAX_RC_TEFF_NORM = (MAX_RC_TEFF - TEFF_MEAN)/TEFF_STD
	MIN_RC_TEFF_NORM = (MIN_RC_TEFF - TEFF_MEAN)/TEFF_STD
	MAX_RC_LOGG_NORM = (MAX_RC_LOGG - LOGG_MEAN)/LOGG_STD
	MIN_RC_LOGG_NORM = (MIN_RC_LOGG - LOGG_MEAN)/LOGG_STD

	'''
	Initializes the model with two helper models, ApogeeNN for Teff, logg regression and ImprovedNN for classification and one spectra embedder model
	'''
	def __init__(self):
		# Load the spectra embedder for dimensionality reduction
		self.spectra_embedder = NormalizedSpectralEmbeddingPCA(E_D =50)
		self.spectra_embedder.load()

		# Load regression and classification models
		self.apogee_nn = load_model(Candle.APOGEE_MODEL_FP)
		self.improved_nn = load_model(Candle.IMPROVED_NN_MODEL_FP)

	'''
	Preprocess a matrix of stellar data from APOGEE DR14 by normalizing it then dimensionally reducting it
	Inputs:
		X - Data matrix of stars, (N, 7514)
	Outputs:
		(N, 50) - normalized and dimensionally reduced stars
	'''
	def preprocess(self, X):
		return self.spectra_embedder.embed(X)


	'''
	Predicts probabilites of sample of stars and returns an array of probabilities, ≥ 0.5 implies the star is a RC
	Inputs:
		X - Data matrix of stars, (N, 7514)
	Outputs:
		y - Vector of probabilities, R^n
	'''
	def predict(self, X):
		N = len(X)

		# Normalize and dimensionally reduce
		X = self.preprocess(X)

		# Regress on teff, logg
		teff, logg = self.apogee_nn.predict(X)

		# Probabilities to return, default is 0 if star is not inside cut
		probabilities = np.zeros((N,))

		# Apply the cut - these are the areas the network knows the best and the theoretical areas of RCs
		desired_idx = [i for i in range(0, N) if teff[i] <= Candle.MAX_RC_TEFF_NORM and teff[i] >= Candle.MIN_RC_TEFF_NORM and logg[i] <= Candle.MAX_RC_LOGG_NORM and logg[i] >= Candle.MIN_RC_LOGG_NORM]

		# Apply the predictions on desired_idx
		probabilities[desired_idx] = self.improved_nn.predict(X[desired_idx]).flatten()

		# Return probabilities
		return probabilities

	'''
	Classifies of sample of stars and returns an array of {0, 1}, 1 is RC
	Inputs:
		X - Data matrix of stars, (N, 7514)
	Outputs:
		y - Vector of {0, 1}, R^n
	'''
	def clasify(self, X):
		N = len(X)
		probabilites = self.predict(X)
		return np.array([1 if probabilites[i] >= 0.5 else 0 for i in range(0, N)])

	'''
	Given a sample of red clump stars, returns the purity of the sample (percentage of stars in the sample that are RC)
	Inputs:
		X - Data matrix of stars, (N, 7514)
	Outputs:
		purity - The percentage of stars that are RC
	'''
	def purity(self):
		pass
