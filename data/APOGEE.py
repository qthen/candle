from astroNN.apogee import allstar
from astropy.io import fits
from astroNN.apogee import combined_spectra, apogee_continuum, visit_spectra
from astroNN.apogee.chips import gap_delete
from data.AstroData import AstroData
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

# Constants about this dataset
MAX_TEFF = 5400
MIN_TEFF = 3500
MAX_LOGG = 4.5
MIN_LOGG = 0

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
	Inputs:
		N - sample size
		dr: Which release to get
		spectra_format: Format of spectra, combined|visit (combined and visit possible for dr13, combined only possible for dr14)
		normalize: Whether or not to apply a normalization to it (only applicable if spectra_format is visit)
		bitmask_value: None (only applicable if spectraFormat is visit and dr is 13)
		save_sample: String - File path to save the sample to
	'''
	def grab_sample(self, N = 32, dr=14, spectra_format='visit', bitmask_value =1, save_sample=True):
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
			'Ni'        : [],
		}

		# Get random indices
		idx = np.random.choice(len(self.apogee_data[1].data['logg']), len(self.apogee_data[1].data['logg']))

		for i in idx:
			# Break condition
			if len(stars['logg']) >= N:
				break

			# For when the spectra_format is visit, we require more than NVISIT >= 1
			if spectra_format == 'visit' and (self.apogee_data[1].data['NVISITS'][i] < 1):
				continue

			# Enforce we have high quality spectra from dr14
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
					star_spectra = None

					# Populating the appropiate star_spectra variable with stellar spectra
					if spectra_format == 'combined':
						local_path_to_file_for_star = combined_spectra(dr=dr, location=location_id, apogee=apogee_id)
						if local_path_to_file_for_star:
							spectra_data = fits.open(local_path_to_file_for_star)
							star_spectra = spectra_data[3].data.copy()
							star_spectra = gap_delete(spectra, dr=dr)
							star_spectra = star_spectra.flatten()
							# Close file handlers
							del spectra_data
					elif spectra_format == 'visit':
						local_path_to_file_for_star = visit_spectra(dr=dr, location=location_id, apogee=apogee_id)
						if local_path_to_file_for_star:
							spectra_data = fits.open(local_path_to_file_for_star)
							spectra, error_spectra, mask_spectra = None, None, None

							# NVISITS is 1, 1D array
							if self.apogee_data[1].data['NVISITS'][i] == 1:
								spectra = spectra_data[1].data
								error_spectra = spectra_data[2].data
								mask_spectra = spectra_data[3].data
							elif self.apogee_data[1].data['NVISITS'][i] > 1:
								spectra = spectra_data[1].data[1]
								error_spectra = spectra_data[2].data[1]
								mask_spectra = spectra_data[3].data[1]

							# Mask if the mask value is present
							if bitmask_value is not None:
								norm_spec, norm_spec_err = apogee_continuum(spectra, error_spectra, cont_mask=None, deg=2, bitmask=mask_spectra, target_bit=None, mask_value=bitmask_value)
								star_spectra = norm_spec.flatten()
							else:
								norm_spec, norm_spec_err = apogee_continuum(spectra, error_spectra, cont_mask=None, deg=2, bitmask=None, target_bit=None, mask_value=None)
								star_spectra = norm_spec.flatten()
							# Close file handlers
							del spectra_data
					else:
						raise ValueError("spectra_format must either be combined|visit")

					# Adding the star data into the dict
					if star_spectra is not None:
						stars['spectra'].append(star_spectra)
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

		# Convert to np style arrays
		for key in stars:
			stars[key] = np.array(stars[key])
		# stars['logg']    = np.array(stars['logg'])
		# stars['Teff']    = np.array(stars['Teff'])
		# stars['M_H']     = np.array(stars['M_H'])
		# stars['C']       = np.array(stars['C'])
		# stars['N']       = np.array(stars['N'])
		# stars['O']       = np.array(stars['O'])
		# stars['Na']      = np.array(stars['Na'])
		# stars['Mg']      = np.array(stars['Mg'])
		# stars['Al']      = np.array(stars['Al'])
		# stars['Si']      = np.array(stars['Si'])
		# stars['P']       = np.array(stars['P'])
		# stars['S']       = np.array(stars['S'])
		# stars['K']       = np.array(stars['K'])
		# stars['Ca']      = np.array(stars['Ca'])
		# stars['Ti']      = np.array(stars['Ti'])
		# stars['V']       = np.array(stars['V'])
		# stars['Mn']      = np.array(stars['Mn'])
		# stars['Fe']      = np.array(stars['Fe'])
		# stars['Ni']      = np.array(stars['Ni'])
		# stars['spectra'] = np.array(stars['spectra'])

		# Save sample if asked to
		if save_sample:
			pickle_out = open("{}apogee_sample_{}_stars.pickle".format(self._PICKLE_DIR, N), 'wb')
			pickle.dump(stars, pickle_out)
			pickle_out.close()

		return stars

	'''
	Sample from APOGEE data based on certain attributes, otherwise randomly sample
	Inputs:
		N - The number of stars to return (max)
		logg_predicate - Lambda function
		Teff_predicate - Lambda function
		normalize - True if we want to normalze
	Returns:
		Sample of APOGEE data in a dict of attributes:
		{
			spectra: [ndarray]
			Teff: [ndarray]
			logg: [ndarray]
		}
	'''
	def get_data(self, N = 100, logg_predicate = lambda x: True, Teff_predicate = lambda x: True, normalize = True):
		self.create_data(normalize = normalize)
		# idx = np.random.choice(len(self.apogee_data[1].data['logg']), len(self.apogee_data[1].data['logg']))
		# stars = {
		# 	'spectra': [],
		# 	'Teff': [],
		# 	'logg': []
		# }
		# for i in idx:
		# 	logg, Teff = self.apogee_data[1].data['logg'][i], self.apogee_data[1].data['Teff'][i]
		# 	if logg_predicate(logg) and Teff_predicate(Teff) and logg != -9999 and Teff != -9999:
		# 		# Passed the predicates, put into stars
		# 		apogee_id, location_id = self.apogee_data[1].data['APOGEE_ID'][i], self.apogee_data[1].data['LOCATION_ID'][i]
		# 		local_path_to_file_for_star = combined_spectra(dr=14, location=location_id, apogee=apogee_id)
		# 		# Valid data
		# 		if local_path_to_file_for_star:
		# 			# Adding spectra data
		# 			spectra_data = fits.open(local_path_to_file_for_star)
		# 			spectra = spectra_data[3].data
		# 			spectra_no_gap = gap_delete(spectra, dr=14)
		# 			spectra_no_gap = spectra_no_gap.flatten()
		# 			stars['spectra'].append(spectra_no_gap)

		# 			# Adding stellar data
		# 			stars['Teff'].append(Teff)
		# 			stars['logg'].append(logg)
				
		# 			# Get rid of file handler
		# 			del spectra_data
		# 			del spectra

		# 	# Break condition
		# 	if len(stars['Teff']) >= N:
		# 		break

		# stars['spectra'] = np.array(stars['spectra'])
		# stars['Teff'] = np.array(stars['Teff'])
		# stars['logg'] = np.array(stars['logg'])
		return stars






