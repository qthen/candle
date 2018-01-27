from data.AstroData import AstroData

'''
Data from APOKASC catalog of Kepler red giants, child of AstroData
'''
class APOKASC(AstroData):

	'''
	Constructor for the class, initializing caching variables and the singleton _star_dict
	'''
	def __init__(self):
		self._star_dict = None
		self._PICKLE_DIR = "pickles/APOKASC/"

	def create_data(self, version = 1):
		raise NotImplementedError("APOKASC data sampler not yet supported")

	def get_data(self, version = 1, max_number_of_stars = float("inf"), use_steps = False, standardize = True, show_data_statistics = True):
		raise NotImplementedError("APOKASC data sampler not yet supported")

