from astroNN.apogee import allstar
from astropy.io import fits
from astroNN.apogee import combined_spectra
from astroNN.apogee.chips import gap_delete
from data.AstroData import AstroData
import numpy as np
import pickle
import os

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
		self._PICKLE_DIR = "pickles/APOGEE/"

	'''
	Sample from the APOGEE data randommly and return a bunch of spectra along with high dimensional labels similar to the Cannon.
	If use_cache is True, then returns a sample from the cached stars we grabbed already
	'''
	def grab_sample(self, N = 32, use_cache = True):
		self.create_data()
		stars = {
			'APSTAR_ID' : [],

			# Spectra of the star
			'spectra'   : [],

			# Stellar features
			'Teff'      : [],
			'logg'      : [],

			# Metals
			'M_H'       : [],
			'C'         : [],
			'N'         : [],
			'O'         : [],
			'Na'        : [],
			'Mg'        : [],
			'Al'        : [],
			'Si'        : [],
			'P'         : [],
			'S'         : [],
			'K'         : [],
			'Ca'        : [],
			'Ti'        : [],
			'V'         : [],
			'Mn'        : [],
			'Fe'        : [],
			'Ni'        : []
		}

		if use_cache:
			if os.path.exists('{}apogee_sample.pickle'.format(self._PICKLE_DIR)):
				pickle_out = open("{}apogee_sample.pickle".format(self._PICKLE_DIR), 'rb')
				stars = pickle.load(pickle_out)

		idx = np.random.choice(len(self.apogee_data[1].data['logg']), len(self.apogee_data[1].data['logg']))
		
		for i in idx:
			# Enforce we have high quality spectra
			if not(self.apogee_data[1].data['STARFLAG'][i] == 0 and self.apogee_data[1].data['ASPCAPFLAG'][i] == 0 and self.apogee_data[1].data['SNR'][i] < 200):

				# Stellar features
				logg = self.apogee_data[1].data['logg'][i]
				Teff = self.apogee_data[1].data['Teff'][i]

				# Metals/H
				M_H  = self.apogee_data[1].data['M_H'][i]

				# Metals
				C    = self.apogee_data[1].data['X_H'][i][0]
				N_c  = self.apogee_data[1].data['X_H'][i][2]
				O    = self.apogee_data[1].data['X_H'][i][3]
				Na   = self.apogee_data[1].data['X_H'][i][4]
				Mg   = self.apogee_data[1].data['X_H'][i][5]
				Al   = self.apogee_data[1].data['X_H'][i][6]
				Si   = self.apogee_data[1].data['X_H'][i][7]
				P    = self.apogee_data[1].data['X_H'][i][8]
				S    = self.apogee_data[1].data['X_H'][i][9]
				K    = self.apogee_data[1].data['X_H'][i][10]
				Ca   = self.apogee_data[1].data['X_H'][i][11]
				Ti   = self.apogee_data[1].data['X_H'][i][12]
				V    = self.apogee_data[1].data['X_H'][i][14]
				Mn   = self.apogee_data[1].data['X_H'][i][16]
				Fe   = self.apogee_data[1].data['X_H'][i][17]
				Ni   = self.apogee_data[1].data['X_H'][i][19]

				# Make sure all of them are not falled
				if all([False if i == -9999 else True for i in [logg, Teff, M_H, C, N_c, O, Na, Mg, Al, Si, P, S, K, Ca, Ti, V, Mn, Fe, Ni]]):

					# Get spectra data
					apogee_id, location_id = self.apogee_data[1].data['APOGEE_ID'][i], self.apogee_data[1].data['LOCATION_ID'][i]
					local_path_to_file_for_star = combined_spectra(dr=14, location=location_id, apogee=apogee_id)
					if local_path_to_file_for_star:
						# Adding spectra data
						spectra_data = fits.open(local_path_to_file_for_star)
						spectra = spectra_data[3].data
						spectra_no_gap = gap_delete(spectra, dr=14)
						spectra_no_gap = spectra_no_gap.flatten()
						stars['spectra'].append(spectra_no_gap)

						# Get rid of file handler
						del spectra_data
						del spectra

						stars['logg'].append(logg)
						stars['Teff'].append(Teff)
						stars['M_H'].append(M_H)
						stars['C'].append(C)
						stars['N'].append(N_c)
						stars['O'].append(O)
						stars['Na'].append(Na)
						stars['Mg'].append(Mg)
						stars['Al'].append(Al)
						stars['Si'].append(Si)
						stars['P'].append(P)
						stars['S'].append(S)
						stars['K'].append(K)
						stars['Ca'].append(Ca)
						stars['Ti'].append(Ti)
						stars['V'].append(V)
						stars['Mn'].append(Mn)
						stars['Fe'].append(Fe)
						stars['Ni'].append(Ni)
						if len(stars['logg']) >= N:
							break
		pickle_out = open("{}apogee_sample.pickle".format(self._PICKLE_DIR), 'wb')
		pickle.dump(stars, pickle_out)
		pickle_out.close()
		return stars

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
		idx = np.random.choice(len(self.apogee_data[1].data['logg']), len(self.apogee_data[1].data['logg']))
		stars = {
			'spectra': [],
			'Teff': [],
			'logg': []
		}
		for i in idx:
			logg, Teff = self.apogee_data[1].data['logg'][i], self.apogee_data[1].data['Teff'][i]
			if logg_predicate(logg) and Teff_predicate(Teff) and logg != -9999 and Teff != -9999:
				# Passed the predicates, put into stars
				apogee_id, location_id = self.apogee_data[1].data['APOGEE_ID'][i], self.apogee_data[1].data['LOCATION_ID'][i]
				local_path_to_file_for_star = combined_spectra(dr=14, location=location_id, apogee=apogee_id)
				# Valid data
				if local_path_to_file_for_star:
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

		stars['spectra'] = np.array(stars['spectra'])
		stars['Teff'] = np.array(stars['Teff'])
		stars['logg'] = np.array(stars['logg'])
		return stars






