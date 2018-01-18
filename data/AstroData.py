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
