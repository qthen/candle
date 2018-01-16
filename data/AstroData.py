# The base class for generating astronmic adata

class AstroData(object):

	'''
	Abstract method to implement, gets the data in a representable way
	'''
	def get_data(self):
		raise NotImplementedError("Abstract method must be overridden")