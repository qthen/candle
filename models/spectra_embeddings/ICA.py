# ICA implementation to reduce star spectra
from sklearn.decomposition import FastICA
from models.spectra_embeddings.Embedding import Embedding

'''
Spectra embedding via ICA
'''
class SpectralEmbeddingICA(Embedding):

	'''
	Constructs PCA for dimensionality reduction of the spectral data
	E_D - The embedding dimension
	'''
	def __init__(self, E_D = 10):
		self.E_D = E_D
		self.FastICA = FastICA(n_components=E_D)


	'''
	Overriding from the Embedding class
	'''
	def fit(self, X):
		self.FastICA.fit(X)


	'''
	Overriding from the Embedding class
	'''
	def embed(self, X): 
		return self.FastICA.transform(X)		
		