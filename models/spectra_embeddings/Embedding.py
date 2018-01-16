# Abstract class for all spectral embedding

class Embedding:

	'''
	Fit model on the embedding
	(m x n) -> (m x e) where e is the embedding dimension and n is the full dimension
	'''
	def fit(self, X):
		raise NotImplementedError("Abstract method needs to be overriden")

	'''
	Embeds the star spectra from a given matrix
	(m x n) -> (m x e) where e is the embedding dimension and n is the full dimension
	'''
	def embed(self, X):
		raise NotImplementedError("Abstract method needs to be overriden")