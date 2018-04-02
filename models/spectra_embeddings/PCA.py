# PCA implementation to reduce star spectra
from sklearn.decomposition import PCA
from models.spectra_embeddings.Embedding import Embedding
import pickle

'''
Spectra embedding via PCA
'''
class SpectralEmbeddingPCA(Embedding):

	MODEL_FILEPATH = "models/spectra_embeddings/PCA.pickle"

	'''
	Constructs PCA for dimensionality reduction of the spectral data
	E_D - The embedding dimension
	'''
	def __init__(self, E_D = 10):
		self.E_D = E_D
		self.PCA = PCA(n_components=E_D)


	'''
	Overriding from the Embedding class
	'''
	def fit(self, X):
		self.PCA.fit(X)


	'''
	Overriding from the Embedding class
	'''
	def embed(self, X): 
		return self.PCA.transform(X)

	'''
	Saves the current PCA trained into a pickle
	'''
	def save(self):
		pickle_out = open(SpectralEmbeddingPCA.MODEL_FILEPATH, 'wb')
		pickle.dump(self.PCA, pickle_out)
		pickle_out.close()

	'''
	Load the PCA using a pickle
	'''
	def load(self):
		pickle_out = open(SpectralEmbeddingPCA.MODEL_FILEPATH, 'rb')
		self.PCA = pickle.load(pickle_out)


		