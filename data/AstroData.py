'''
The base class for getting astronomy data for training models. Objects inheriting from AstroData should
preform some data preparation and return meaningful objects (structs) representing useful data for training models
which in general should have the form of:
{
	spectra: [SPECTRA]
	ps: [Float]
	Dnu: [Float]
	...
}
PS is the period spacing observed in the star, s
Δv is Dnu - large frequency separation, muHz
spectra: Spectra data from APOGEE

All children inheriting from AstroData should store data in a singleton and have two methods:
create_data: Create the actual data and save it in whatever suitable form (usually a table and a pickle) then populate singleton
get_data: Return singleton of data

Data may have many versions, read from the pickles/data_wanted README to figure out which one is desired

Process of data curation typically requires dependencies of partial data from /table then pickling the completed dict into /pickles

AstroData also provides the method, is_red_clump(ps, Dnu) which takes stellar parameters and returns a 0 or 1. This is based on the Vrad, 2016 paper
'''
class AstroData(object):

	'''
	Abstract method to implement, gets the data in a representable way, pickles it for caching then returns it
	'''
	def get_data(self):
		raise NotImplementedError("Abstract method must be overridden")

	'''
	Abstract method to implement, creates the data in a representable way and saves it into a table and/or pickle
	'''
	def create_data(self):
		raise NotImplementedError("Abstract method must be overridden")

	'''
	Based on Vrad, 2016 paper on Period Spacings in Red Giants II, a line of best fit was drawn on a plot of red giants by Δv and PS and acts as the best ground truth for red clump classification
	Inputs:
		PS - [Float], the period spacing of a star
		Dnu - [Float], the Δv of a star, large frequency separation
	Returns:
		1|0 - Float, 1 denotes red clump status
	'''
	def is_red_clump(self, PS, Dnu):
		return 1.0 if PS >= ((float(75)/float(26))*Dnu + 100.0) else 0.0