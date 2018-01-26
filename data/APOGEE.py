from astroNN.apogee import allstar
from astropy.io import fits
from astroNN.apogee import combined_spectra
from astroNN.apogee.chips import gap_delete
from data.AstroData import AstroData
import numpy as np

'''
Helper class to sample data from APOGEE spectra data, inherits from AstroData
'''
class APOGEE(AstroData):

	'''
	Helper function to create the data to sample from
	'''
	def create_data(self):
		local_path_to_file = allstar(dr=14)
		self.apogee_data = fits.open(local_path_to_file)


	'''
	Sample from APOGEE data based on certain attributes, otherwise randomly sample
	Inputs:
		N - The number of stars to return (max)
		logg_predicate - Lambda function
		Teff_predicate - Lambda function
	Returns:
		Sample of APOGEE data in a dict of attributes:
		{
			spectra: [ndarray]
			Teff: [ndarray]
			logg: [ndarray]
		}
	'''
	def get_data(self, N = 100, logg_predicate = lambda x: True, Teff_predicate = lambda x: True):
		self.create_data()
		idx = np.random.choice(29502, 29502) # 29,502 objects in APOGEE dr=14
		stars = {
			'spectra': [],
			'Teff': [],
			'logg': []
		}
		for i in idx:
			logg, Teff = self.apogee_data[1].data['logg'][i], self.apogee_data[1].data['Teff'][i]
			if logg_predicate(logg) and Teff_predicate(Teff):
				# Passed the predicates, put into stars
				apogee_id, location_id = self.apogee_data[1].data['APOGEE_ID'][i], self.apogee_data[1].data['LOCATION_ID'][i]
				local_path_to_file_for_star = combined_spectra(dr=14, location=location_id, apogee=apogee_id)

				# Valid data
				if local_path_to_file_for_star and logg != -9999 and Teff != -9999:
					# Adding spectra data
					spectra_data = fits.open(local_path_to_file_for_star)
					spectra = spectra_data[3].data
					spectra_no_gap = gap_delete(spectra, dr=14)
					spectra_no_gap = spectra_no_gap.flatten()
					stars['spectra'].append(spectra_no_gap)

					# Adding stellar data
					stars['Teff'].append(Teff)
					stars['logg'].append(logg)
				
					# Get rid of file handler
					del spectra_data
					del spectra

			# Break condition
			if len(stars['Teff']) >= N:
				break
		return stars






