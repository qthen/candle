from astropy.table import Table
from astropy.table import Column
from astroNN.apogee import allstar
from astropy.io import fits
from astroNN.datasets.xmatch import xmatch
from astroNN.apogee import combined_spectra, visit_spectra, apogee_continuum
from astroNN.apogee.chips import gap_delete
from data.AstroData import AstroData
from sklearn import preprocessing
import pickle
import numpy as np
import os.path
import matplotlib.pyplot as plt

# Constants about this Kepler dataset
MAX_TEFF = 5200
MIN_TEFF = 4500
MAX_LOGG = 3.5
MIN_LOGG = 2.25

'''
AstroData child for generating and handling data for Kepler stars observed with PS and Δv
Retrieves data from `Vrad 2016` table and xmatch's for spectra
'''
class KeplerPeriodSpacing(AstroData):

	'''
	Constructor for the class, initializing caching variables and the singleton _star_dict
	'''
	def __init__(self):
		self._star_dict = None
		self._star_version = None
		self._PICKLE_DIR = "pickles/kepler/"


	'''
	Helper function for populating original KIC catalog of red giants and their asteroseismic parameters with RA and DEC
	RA and DEC is populated from "tables/KIC_A87/data_0.csv"
	Newly populated table with RA, DEC is written to "tables/KIC_A87/table1.dat" and explained in README

	This is because the original table: Period spacings in red giants. II. Automated measurement. doesn't come with RA and DEC columns
	so I need to match their KIC with the Kepler Catalog that contains RA and DEC

	Returns:
		Boolean - True on success of writing the new table, false or throws otherwise
	'''
	def _populate_kepler_with_rc_dec(self):
		# Creating the KIC_A87/table1.dat file which contains all the KIC stars with their RA and DEC
		# If this already exists, we can load it from the tables/ file and proceed immediately to get the APOGEE spectra
		if not os.path.exists('tables//KIC_A87/table1.dat'):

			# This gets me the PS and astroseismic parameters from the KIC, but not the RA or DE
			data_ps = Table.read("tables/A87/table2.dat", format="ascii")

			# This gets me the RA and DE of the stars above into a dict
			data_ra_de = {}
			with open("tables/KIC_A87/data_0.csv") as fp:
				skip = False
				for line in fp:
					if not skip:
						skip = True
						continue
					kic, ra, de = line.split(",")[1:4]
					data_ra_de[int(kic)] = (float(ra), float(de))

			# Merge the two tables together with new columns: RA and DE inside data_ps
			data_ps_array = data_ps.as_array()
			ra_arr = []
			de_arr = []
			for i in range(0, len(data_ps_array)):
				ra_arr.append(data_ra_de[int(data_ps_array[i][0])][0])
				de_arr.append(data_ra_de[int(data_ps_array[i][0])][1])
			ra_column = Column(name='RA', data=ra_arr)
			de_column = Column(name='DE', data=de_arr)
			data_ps.add_columns((ra_column, de_column))

			# Renaming for help
			data_ps.rename_column('col1', 'KIC')
			data_ps.rename_column('col2', 'Dnu')
			data_ps.rename_column('col3', 'DPi1')
			data_ps.rename_column('col4', 'e_DPi1')
			data_ps.rename_column('col5', 'q')
			data_ps.rename_column('col6', 'M')
			data_ps.rename_column('col7', 'e_M')
			data_ps.rename_column('col8', 'Alias')
			data_ps.rename_column('col9', 'Measure')
			data_ps.rename_column('col10', 'Status')

			data_ps.write('../tables/KIC_A87/table1.dat' , format='ascii')
		return True


	'''
	Overriding from AstroData, creates table KIC_A87/table1.dat that has KIC stars with RA, DEC
	Then matches with APOGEE data and pickles into pickles/kepler/stars1.pickle
	Populates singleton, self._star_dict
	Inputs:
		version - The version of the data to use (1 = stars1.pickle, etc.)
		max_number_of_stars - Maximum number of star data to return, by default returns all
		use_steps - If True then returns spectra that is usable for convolutional networks (batch_size, steps, 1)
	Returns:
		Boolean - True on success, false or throws otherwise
	'''
	def create_data(self, version = 1, max_number_of_stars = float("inf"), use_steps = False):

		if os.path.exists('{}stars{}.pickle'.format(self._PICKLE_DIR, version)):
			pickle_out = open("{}stars{}.pickle".format(self._PICKLE_DIR, version), 'rb')
			self._star_dict = pickle.load(pickle_out)
			return True

		self._star_dict = {
			'KIC'     : [],
			'RA'      : [],
			'DEC'     : [],
			'Dnu'     : [],
			'PS'      : [],
			'e_PS'    : [],
			'q'       : [],
			'M'       : [],
			'spectra' : [],
			'error_spectra': [],
			'T_eff'   : [],
			'logg'    : [],
			'RC'      : [],
			'status'  : [],
			'M_H'     : [],
			'C'       : [],
			'N'       : [],
			'O'       : [],
			'Na'      : [],
			'Mg'      : [],
			'Al'      : [],
			'Si'      : [],
			'P'       : [],
			'S'       : [],
			'K'       : [],
			'Ca'      : [],
			'Ti'      : [],
			'V'       : [],
			'Mn'      : [],
			'Fe'      : [],
			'Ni'      : [],
			'Cr'      : [],
			'Co'      : [],
			'Cl'      : []
		}

		# First create table of Kepler stars with PS and Δv with their RA, DEC (creates table1.dat)
		self._populate_kepler_with_rc_dec()

		# Loading in APOGEE star data (RA, DEC)
		local_path_to_file = allstar(dr=14)
		apogee_data = fits.open(local_path_to_file)
		apogee_ra = apogee_data[1].data['RA']
		apogee_de = apogee_data[1].data['DEC']

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

			if version == 1 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and apogee_data[1].data['ASPCAPFLAG'][index_in_apogee] == 0):
				continue
			
			# Version 2 - Subject to the following constraints
			# Flag checking, condition is the same as Hawkins et al. 2017, STARFLAG and ASPCAP bitwise OR is 0
			# KIC checking, uncertainty in PS should be less than 10s
			if version == 2 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and apogee_data[1].data['ASPCAPFLAG'][index_in_apogee] == 0 and kic_table['e_DPi1'][index_in_kepler] < 10):
					continue

			# Version 3 - Subject to the following constraints
			# Flag checking, condition is the same as Hawkins et al. 2017, STARFLAG and ASPCAP bitwise OR is 0
			# KIC checking, uncertainty in PS should be less than 10s
			if version == 3 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and apogee_data[1].data['ASPCAPFLAG'][index_in_apogee] == 0 and kic_table['e_DPi1'][index_in_kepler] < 10 and apogee_data[1].data['SNR'][index_in_apogee] >= 200):
					continue

			# Version 5 - DR13 data subject to same connstraints as Hawkings
			if version == 5 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and apogee_data[1].data['ASPCAPFLAG'][index_in_apogee] == 0 and kic_table['e_DPi1'][index_in_kepler] < 10):
					continue

			# Version 6 - DR13 data subject to same connstraints as Hawkings - normalization with bitmask as DR13
			if version == 6 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and apogee_data[1].data['ASPCAPFLAG'][index_in_apogee] == 0 and kic_table['e_DPi1'][index_in_kepler] < 10 and apogee_data[1].data['NVISITS'][index_in_apogee] >= 1):
					continue

			# Version 7 - DR13 data subject to same constraints as Hawkings - no bit mask as DR13
			if version == 7 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and kic_table['e_DPi1'][index_in_kepler] < 10 and apogee_data[1].data['NVISITS'][index_in_apogee] >= 1):
					continue

			# Version 8 - DR13 data subject to same constraints as Hawkings
			if version == 8 and not(apogee_data[1].data['STARFLAG'][index_in_apogee] == 0 and kic_table['e_DPi1'][index_in_kepler] < 10 and apogee_data[1].data['NVISITS'][index_in_apogee] >= 1):
					continue

			a_apogee_id = apogee_data[1].data['APOGEE_ID'][index_in_apogee]
			a_location_id = apogee_data[1].data['LOCATION_ID'][index_in_apogee]

			
			local_path_to_file_for_star = None
			if version == 6 or version == 7 or version == 8:
				local_path_to_file_for_star = visit_spectra(dr=14, location=a_location_id, apogee=a_apogee_id)
			elif version == 4 or version == 5:
				local_path_to_file_for_star = combined_spectra(dr=13, location=a_location_id, apogee=a_apogee_id)
			else:
				local_path_to_file_for_star = combined_spectra(dr=14, location=a_location_id, apogee=a_apogee_id)

			if (local_path_to_file_for_star):
				# Filter out the dr=14 gaps, value to be appended to the array
				spectra_no_gap = None
				error_spectra = None
				if version in (6, 7, 8):
					# Read from visit spectra of the star
					spectra_data = fits.open(local_path_to_file_for_star)
					spectra, mask_spectra = None, None

					# Best fit spectra data - use for spectra data
					if apogee_data[1].data['NVISITS'][index_in_apogee] == 1:
						spectra = spectra_data[1].data
						error_spectra = spectra_data[2].data
						mask_spectra = spectra_data[3].data
					elif apogee_data[1].data['NVISITS'][index_in_apogee] > 1:
						spectra = spectra_data[1].data[1]
						error_spectra = spectra_data[2].data[1]
						mask_spectra = spectra_data[3].data[1]

					if version == 6:
						norm_spec, norm_spec_err = apogee_continuum(spectra, error_spectra, cont_mask=None, deg=2, dr=13, bitmask=mask_spectra, target_bit=None)
						spectra_no_gap = norm_spec
						error_spectra = norm_spec_err
					elif version == 7:
						norm_spec, norm_spec_err = apogee_continuum(spectra, error_spectra, cont_mask=None, deg=2, dr=13, bitmask=None, target_bit=None)
						spectra_no_gap = norm_spec
						error_spectra = norm_spec_err
					elif version == 8:
						# Bit mask value is now 1
						norm_spec, norm_spec_err = apogee_continuum(spectra, error_spectra, cont_mask=None, deg=2, dr=13, bitmask=mask_spectra, target_bit=None, mask_value=1)
						spectra_no_gap = norm_spec
						error_spectra = norm_spec_err
					del mask_spectra

				elif version == 4 or version == 5:
					spectra_data = fits.open(local_path_to_file_for_star)

					# Best fit spectra data - use for spectra data
					spectra = spectra_data[3].data
					spectra_no_gap = gap_delete(spectra, dr=13)
				else:
					spectra_data = fits.open(local_path_to_file_for_star)

					# Best fit spectra data - use for spectra data
					spectra = spectra_data[3].data

					spectra_no_gap = gap_delete(spectra, dr=14)

				spectra_no_gap = spectra_no_gap.flatten()

				# APOGEE data
				self._star_dict['T_eff'].append(apogee_data[1].data['Teff'][index_in_apogee].copy())
				self._star_dict['logg'].append(apogee_data[1].data['logg'][index_in_apogee].copy())

				# Metals/H
				self._star_dict['M_H'].append(apogee_data[1].data['M_H'][index_in_apogee])
				self._star_dict['C'].append(apogee_data[1].data['X_H'][index_in_apogee][0])
				self._star_dict['N'].append(apogee_data[1].data['X_H'][index_in_apogee][2])
				self._star_dict['O'].append(apogee_data[1].data['X_H'][index_in_apogee][3])
				self._star_dict['Na'].append(apogee_data[1].data['X_H'][index_in_apogee][4])
				self._star_dict['Mg'].append(apogee_data[1].data['X_H'][index_in_apogee][5])
				self._star_dict['Al'].append(apogee_data[1].data['X_H'][index_in_apogee][6])
				self._star_dict['Si'].append(apogee_data[1].data['X_H'][index_in_apogee][7])
				self._star_dict['P'].append(apogee_data[1].data['X_H'][index_in_apogee][8])
				self._star_dict['S'].append(apogee_data[1].data['X_H'][index_in_apogee][9])
				self._star_dict['K'].append(apogee_data[1].data['X_H'][index_in_apogee][10])
				self._star_dict['Ca'].append(apogee_data[1].data['X_H'][index_in_apogee][11])
				self._star_dict['Ti'].append(apogee_data[1].data['X_H'][index_in_apogee][12])
				self._star_dict['V'].append(apogee_data[1].data['X_H'][index_in_apogee][14])
				self._star_dict['Mn'].append(apogee_data[1].data['X_H'][index_in_apogee][16])
				self._star_dict['Fe'].append(apogee_data[1].data['X_H'][index_in_apogee][17])
				self._star_dict['Ni'].append(apogee_data[1].data['X_H'][index_in_apogee][19])
				self._star_dict['Cr'].append(apogee_data[1].data['X_H'][index_in_apogee][15])
				self._star_dict['Co'].append(apogee_data[1].data['X_H'][index_in_apogee][18])
				self._star_dict['Cl'].append(apogee_data[1].data['X_H'][index_in_apogee][1])

				# KIC data
				self._star_dict['KIC'].append(kic_table['KIC'][index_in_kepler])
				self._star_dict['RA'].append(kic_table['RA'][index_in_kepler])
				self._star_dict['DEC'].append(kic_table['DE'][index_in_kepler])
				self._star_dict['Dnu'].append(kic_table['Dnu'][index_in_kepler])
				self._star_dict['PS'].append(kic_table['DPi1'][index_in_kepler])
				self._star_dict['e_PS'].append(kic_table['e_DPi1'][index_in_kepler])
				self._star_dict['q'].append(kic_table['q'][index_in_kepler])
				self._star_dict['M'].append(kic_table['M'][index_in_kepler])
				self._star_dict['status'].append(kic_table['Status'][index_in_kepler])
				self._star_dict['RC'].append(self.is_red_clump(kic_table['DPi1'][index_in_kepler], kic_table['Dnu'][index_in_kepler]))

				# Gap delete doesn't return row vector, need to manually reshape
				self._star_dict['spectra'].append(spectra_no_gap)

				if version in (6, 7, 8):
					self._star_dict['error_spectra'].append(error_spectra.flatten())
					del error_spectra

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
		self._star_dict['e_PS']    = np.array(self._star_dict['e_PS'])
		self._star_dict['spectra'] = np.array(self._star_dict['spectra'])
		self._star_dict['logg']    = np.array(self._star_dict['logg'])
		self._star_dict['T_eff']   = np.array(self._star_dict['T_eff'])
		self._star_dict['RC']      = np.array(self._star_dict['RC'])
		self._star_dict['status']  = np.array(self._star_dict['status'])
		self._star_dict['M']       = np.array(self._star_dict['M'])
		self._star_dict['M_H']     = np.array(self._star_dict['M_H'])
		self._star_dict['C']       = np.array(self._star_dict['C'])
		self._star_dict['N']       = np.array(self._star_dict['N'])
		self._star_dict['O']       = np.array(self._star_dict['O'])
		self._star_dict['Na']      = np.array(self._star_dict['Na'])
		self._star_dict['Mg']      = np.array(self._star_dict['Mg'])
		self._star_dict['Al']      = np.array(self._star_dict['Al'])
		self._star_dict['Si']      = np.array(self._star_dict['Si'])
		self._star_dict['P']       = np.array(self._star_dict['P'])
		self._star_dict['S']       = np.array(self._star_dict['S'])
		self._star_dict['K']       = np.array(self._star_dict['K'])
		self._star_dict['Ca']      = np.array(self._star_dict['Ca'])
		self._star_dict['Ti']      = np.array(self._star_dict['Ti'])
		self._star_dict['V']       = np.array(self._star_dict['V'])
		self._star_dict['Mn']      = np.array(self._star_dict['Mn'])
		self._star_dict['Fe']      = np.array(self._star_dict['Fe'])
		self._star_dict['Ni']      = np.array(self._star_dict['Ni'])
		self._star_dict['Cl']      = np.array(self._star_dict['Cl'])
		self._star_dict['Cr']      = np.array(self._star_dict['Cr'])
		self._star_dict['Co']      = np.array(self._star_dict['Co'])
		self._star_dict['q']       = np.array(self._star_dict['q'])

		if version == 6:
			self._star_dict['error_spectra'] = np.array(self._star_dict['error_spectra'])

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

		if not self._star_dict or self._star_version != version:
			self.create_data(version, max_number_of_stars, use_steps)

			if show_data_statistics:
				# Print information about the data
				print(("Kepler period spacing measurements with 𝚫v and APOGEE spectra\n" +
					"-------------------------------------------------------------\n" +
					"Stars: {}\n" +
					"Projected size: {}mb").format(len(self._star_dict['KIC']), len(self._star_dict['KIC'])*9000*64*1.25e-7))

		if standardize:
			# Star spectra seems tricky to standardize due to numerical issues, should look into this
			self._star_dict['KIC']     = self._star_dict['KIC']
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


