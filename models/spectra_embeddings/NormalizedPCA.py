from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.spectra_embeddings.Embedding import Embedding
import pickle

'''
Spectral embedding that is typically used over standard PCA

First each vector is normalized by axis = 0
Then it is given to PCA to be fit

Typically be used on a large data set (for example, 10K sampled stars from APOGEE)
'''
class NormalizedSpectralEmbeddingPCA(Embedding):

	MODEL_FILEPATH = "models/spectra_embeddings/NormalizedPCA.pickle"
	SCALER_FILEPATH = "models/spectra_embeddings/ScalerPCA.pickle"

	'''
	Constructs PCA for dimensionality reduction of the spectral data
	E_D - The embedding dimension
	'''
	def __init__(self, E_D = 10):
		self.E_D = E_D
		self.scaler = StandardScaler()
		self.PCA = PCA(n_components=E_D)


	'''
	Overriding from the Embedding class
	Inputs:
		X - Matrix of row vectors of spectra
	'''
	def fit(self, X):
		self.scaler.fit(X)
		self.PCA.fit(self.scaler.transform(X))


	'''
	Overriding from the Embedding class
	Inputs:
		X - Matrix of row vectors of spectra
	'''
	def embed(self, X):
		# First normalize them with our scaler
		X = self.scaler.transform(X)

		# Now embed them with PCA
		return self.PCA.transform(X)

	'''
	Saves the current normalized PCA trained into a pickle
	'''
	def save(self):
		pickle_out = open(NormalizedSpectralEmbeddingPCA.MODEL_FILEPATH, 'wb')
		pickle.dump(self.PCA, pickle_out)
		pickle_out.close()

		pickle_out = open(NormalizedSpectralEmbeddingPCA.SCALER_FILEPATH, 'wb')
		pickle.dump(self.scaler, pickle_out)
		pickle_out.close()

	'''
	Load the PCA using a pickle
	'''
	def load(self):
		pickle_out = open(NormalizedSpectralEmbeddingPCA.MODEL_FILEPATH, 'rb')
		self.PCA = pickle.load(pickle_out)

		pickle_out = open(NormalizedSpectralEmbeddingPCA.SCALER_FILEPATH, 'rb')
		self.scaler = pickle.load(pickle_out)


		