
from astropy.table import Table
from astropy.table import Column
from astroNN.apogee import allstar
from astropy.io import fits
from astroNN.datasets.xmatch import xmatch
from astroNN.apogee import combined_spectra
from astroNN.apogee.chips import gap_delete
from data.AstroData import AstroData
from sklearn import preprocessing
import pickle
import numpy as np
import os.path

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

	'''
	Overriding from AstroData, creates the pickle stars in pickles/APOKASC
	Populates singleton, self._star_dict
	Inputs:
		version - The version of the data to use (1 = stars1.pickle, etc.)
	Returns:
		Boolean - True on success, false or throws otherwise
	'''
	def create_data(self, version = 1):

		if os.path.exists('{}stars{}.pickle'.format(self._PICKLE_DIR, version)):
			pickle_out = open("{}stars{}.pickle".format(self._PICKLE_DIR, version), 'rb')
			self._star_dict = pickle.load(pickle_out)
			return True

		# Create the data
		self._star_dict = {
			'KIC'     : [],
			'RA'      : [],
			'DEC'     : [],
			'Dnu'     : [],
			'PS'      : [],
			'spectra' : [],
			'T_eff'   : [],
			'logg'    : [],
			'RC'      : [],
		}

		# First create table of Kepler stars with PS and Δv with their RA, DEC (creates table1.dat)
		self._populate_kepler_with_rc_dec()

		# Loading in APOGEE star data (RA, DEC)
		local_path_to_file = allstar(dr=14)
		apogee_data = fits.open(local_path_to_file)
		apogee_ra = apogee_data[1].data['ra']
		apogee_de = apogee_data[1].data['dec']

		# RA, DEC from KIC table
		kic_table = Table.read("tables/KIC_A87/table1.dat", format="ascii")
		kic_ra = np.array(kic_table['RA'])
		kic_de = np.array(kic_table['DE'])

		# Indices overlap between KIC and APOGEE
		idx_kic, idx_apogee, sep = xmatch(kic_ra, apogee_ra, colRA1=kic_ra, colDec1 = kic_de, colRA2 = apogee_ra, colDec2 = apogee_de)

		# Building spectra data for KIC from APOGEE overlaps
		for i in range(0, len(idx_apogee)):
			index_in_apogee = idx_apogee[i]
			index_in_kepler = idx_kic[i]
			
			# Version 2 - Subject to the following constraints
			# Flag checking, condition is the same as Hawkins et al. 2017, STARFLAG and ASPCAP bitwise OR is 0
			# KIC checking, uncertainty in PS should be less than 10s
			if version == 2 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and apogee_data[1].data['ASPCAPFLAG'][index_in_apogee] == 0 and kic_table['e_DPi1'][index_in_kepler] < 10):
					continue

			a_apogee_id = apogee_data[1].data['APOGEE_ID'][index_in_apogee]
			a_location_id = apogee_data[1].data['LOCATION_ID'][index_in_apogee]

			local_path_to_file_for_star = combined_spectra(dr=14, location=a_location_id, apogee=a_apogee_id)
			if (local_path_to_file_for_star):
				spectra_data = fits.open(local_path_to_file_for_star)

				# Best fit spectra data - use for spectra data
				spectra = spectra_data[3].data

				# Filter out the dr=14 gaps
				spectra_no_gap = gap_delete(spectra, dr=14)
				spectra_no_gap = spectra_no_gap.flatten()

				# APOGEE data
				self._star_dict['T_eff'].append(apogee_data[1].data['Teff'][index_in_apogee].copy())
				self._star_dict['logg'].append(apogee_data[1].data['logg'][index_in_apogee].copy())

				# KIC data
				self._star_dict['KIC'].append(kic_table['KIC'][index_in_kepler])
				self._star_dict['RA'].append(kic_table['RA'][index_in_kepler])
				self._star_dict['DEC'].append(kic_table['DE'][index_in_kepler])
				self._star_dict['Dnu'].append(kic_table['Dnu'][index_in_kepler])
				self._star_dict['PS'].append(kic_table['DPi1'][index_in_kepler])
				self._star_dict['RC'].append(self.is_red_clump(kic_table['DPi1'][index_in_kepler], kic_table['Dnu'][index_in_kepler]))

				# Gap delete doesn't return row vector, need to manually reshape
				self._star_dict['spectra'].append(spectra_no_gap)

				# Close file handler
				del spectra_data
				del spectra

			# Check max condition
			if i > max_number_of_stars - 1:
				break

		# Convert to numpy arrays
		self._star_dict['KIC']     = np.array(self._star_dict['KIC'])
		self._star_dict['RA']      = np.array(self._star_dict['RA'])
		self._star_dict['DEC']     = np.array(self._star_dict['DEC'])
		self._star_dict['Dnu']     = np.array(self._star_dict['Dnu'])
		self._star_dict['PS']      = np.array(self._star_dict['PS'])
		self._star_dict['spectra'] = np.array(self._star_dict['spectra'])
		self._star_dict['logg']    = np.array(self._star_dict['logg'])
		self._star_dict['T_eff']   = np.array(self._star_dict['T_eff'])
		self._star_dict['RC']      = np.array(self._star_dict['RC'])

		# Pickle for caching
		pickle_out = open("{}stars{}.pickle".format(self._PICKLE_DIR, version), 'wb')
		pickle.dump(self._star_dict, pickle_out)
		pickle_out.close()

		return True

	'''
	Given the relevant files for the tables, returns a dict of data points, keys are the variable
	Inputs:
		version - The version of the data to use (1 = stars1.pickle, etc.)
		max_number_of_stars - Maximum number of star data to return, by default returns all
		use_steps - If True then returns spectra that is usable for convolutional networks (batch_size, steps, 1)
		standardize - Standardize all values to be Gaussian centered at 0 with std. = 1
		show_data_statistics - Print the statistics of the loaded stars file
	Returns:
		dict: { 
			KIC     : [Int]
			RA      : [Float]
			DEC     : [Float]
			spectra : [[Float]]
			Dnu     : [Float]
			PS      : [Float]
			T_eff   : [Float]
			logg    : [Float],
			RC      : [Int]
		}
	'''
	def get_data(self, version = 1, max_number_of_stars = float("inf"), use_steps = False, standardize = True, show_data_statistics = True):

		if not self._star_dict:
			self.create_data(version, max_number_of_stars, use_steps)

			if show_data_statistics:
				# Print information about the data
				print(("Kepler period spacing measurements with 𝚫v and APOGEE spectra\n" +
					"-------------------------------------------------------------\n" +
					"Stars: {}\n" +
					"Projected size: {}mb").format(len(self._star_dict['KIC']), len(self._star_dict['KIC'])*9000*64*1.25e-7))

		if standardize:
			# Star spectra seems tricky to standardize due to numerical issues, should look into this
			self._star_dict['KIC']     = preprocessing.scale(self._star_dict['KIC'])
			self._star_dict['RA']      = preprocessing.scale(np.array(self._star_dict['RA']))
			self._star_dict['DEC']     = preprocessing.scale(np.array(self._star_dict['DEC']))
			self._star_dict['Dnu']     = preprocessing.scale(np.array(self._star_dict['Dnu']))
			self._star_dict['PS']      = preprocessing.scale(np.array(self._star_dict['PS']))
			self._star_dict['spectra'] = preprocessing.scale(self._star_dict['spectra'])
			self._star_dict['logg']    = preprocessing.scale(np.array(self._star_dict['logg']))
			self._star_dict['T_eff']   = preprocessing.scale(np.array(self._star_dict['T_eff']))

		# Check if need to use steps - mainly for use for convolutional networks
		if use_steps:
			self._star_dict['spectra'] = self._star_dict['spectra'].reshape((self._star_dict['spectra'].shape[0], self._star_dict['spectra'].shape[1], 1))

		return self._star_dict


