from data.AstroData import AstroData
from astropy.table import Table
from astropy.table import Column
from astroNN.apogee import allstar
from astropy.io import fits
from astroNN.datasets.xmatch import xmatch
from astroNN.apogee import combined_spectra
from astroNN.apogee.chips import gap_delete
import pickle
import numpy as np
import os.path

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
			'spectra' : [],
			'T_eff'   : [],
			'logg'    : [],
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
			a_apogee_id = apogee_data[1].data['APOGEE_ID'][idx_apogee[i]]
			a_location_id = apogee_data[1].data['LOCATION_ID'][idx_apogee[i]]
			local_path_to_file_for_star = combined_spectra(dr=14, location=a_location_id, apogee=a_apogee_id)
			if (local_path_to_file_for_star):
				spectra_data = fits.open(local_path_to_file_for_star)

				# Best fit spectra data - use for spectra data
				spectra = spectra_data[3].data

				# Filter out the dr=14 gaps
				spectra_no_gap = gap_delete(spectra, dr=14)
				spectra_no_gap = spectra_no_gap.flatten()

				# APOGEE data
				self._star_dict['T_eff'].append(apogee_data[1].data['Teff'][idx_apogee[i]].copy())
				self._star_dict['logg'].append(apogee_data[1].data['logg'][idx_apogee[i]].copy())

				# KIC data
				self._star_dict['KIC'].append(kic_table['KIC'][idx_kic[i]])
				self._star_dict['RA'].append(kic_table['RA'][idx_kic[i]])
				self._star_dict['DEC'].append(kic_table['DE'][idx_kic[i]])
				self._star_dict['Dnu'].append(kic_table['Dnu'][idx_kic[i]])
				self._star_dict['PS'].append(kic_table['DPi1'][idx_kic[i]])

				# Gap delete doesn't return row vector, need to manually reshape
				if not use_steps:
					self._star_dict['spectra'].append(spectra_no_gap)
				else:
					# Need to reshape to steps
					self._star_dict['spectra'].append(spectra_no_gap.reshape(spectra_no_gap.shape[0], 1))

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
	Returns:
		dict: { 
			KIC     : [Int]
			RA      : [Float]
			DEC     : [Float]
			spectra : [[Float]]
			Dnu     : [Float]
			PS      : [Float]
			T_eff   : [Float]
			logg    : [Float]
		}
	'''
	def get_data(self, version = 1,max_number_of_stars = float("inf"), use_steps = False):

		if not self._star_dict:
			self.create_data(version, max_number_of_stars, use_steps)

			# Print information about the data
			print(("Kepler period spacing measurements with 𝚫v and APOGEE spectra\n" +
				"-------------------------------------------------------------\n" +
				"Stars: {}\n" +
				"Projected size: {}mb").format(len(self._star_dict['KIC']), len(self._star_dict['KIC'])*9000*64*1.25e-7))

		return self._star_dict
