# PCA implementation to reduce star spectra
from sklearn.decomposition import PCA
from models.spectra_embeddings.Embedding import Embedding

'''
Spectra embedding via PCA
'''
class SpectralEmbeddingPCA(Embedding):

	'''
	Constructs PCA for dimensionality reduction of the spectral data
	E_D - The embedding dimension
	'''
	def __init__(self, E_D):
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
		