# Class for generating 
from data.AstroData import AstroData
from astropy.table import Table
from astropy.table import Column
import numpy as np
import os.path
from astroNN.apogee import allstar
from astropy.io import fits
from astroNN.datasets.xmatch import xmatch
from astroNN.apogee import combined_spectra
from astroNN.apogee.chips import gap_delete

class KeplerPeriodSpacing(AstroData):

	'''
	Given the relevant files for the tables, returns a dict of data points, keys are the variable
	Inputs:
		max_number_of_stars - Maximum number of star data to return, by default returns all
		use_steps - If True then returns spectra that is usable for convolutional networks (batch_size, steps, 1)
	Returns:
		dict: { KIC, RA, DEC, SPECTRA , Dnu, Dpi1 } -> np arrays
	'''
	def get_data(self, max_number_of_stars = float("inf"), use_steps = False):

		# Creating the KIC_A87/table1.dat file which contains all the KIC stars with their RA and DEC
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

		# APOGEE star data (no spectra)
		local_path_to_file = allstar(dr=14)
		apogee_data = fits.open(local_path_to_file)

		# APOGEE celestical coordinates
		apogee_ra = apogee_data[1].data['ra']
		apogee_de = apogee_data[1].data['dec']

		# KIC celestical coordinates
		kic_table = Table.read("tables/KIC_A87/table1.dat", format="ascii")
		kic_ra = np.array(kic_table['RA'])
		kic_de = np.array(kic_table['DE'])

		# Indices overlap between KIC and APOGEE
		idx_kic, idx_apogee, sep = xmatch(kic_ra, apogee_ra, colRA1=kic_ra, colDec1 = kic_de, colRA2 = apogee_ra, colDec2 = apogee_de)

		# Dict to return containing the star data
		star_dict = {
			'KIC'     : [],
			'RA'      : [],
			'DEC'     : [],
			'Dnu'     : [],
			'DPi1'    : [],
			'spectra' : []
		}

		# Building spectra data for KIC from APOGEE
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

				# KIC data
				star_dict['KIC'].append(kic_table['KIC'][idx_kic[i]])
				star_dict['RA'].append(kic_table['RA'][idx_kic[i]])
				star_dict['DEC'].append(kic_table['DE'][idx_kic[i]])
				star_dict['Dnu'].append(kic_table['Dnu'][idx_kic[i]])
				star_dict['DPi1'].append(kic_table['DPi1'][idx_kic[i]])

				# Gap delete doesn't return row vector, need to manually reshape
				if not use_steps:
					star_dict['spectra'].append(spectra_no_gap)
				else:
					# Need to reshape to steps
					star_dict['spectra'].append(spectra_no_gap.reshape(spectra_no_gap.shape[0], 1))

				# Close file handler
				del spectra_data
				del spectra

			# Check max condition
			if i > max_number_of_stars - 1:
				break

		# Convert to numpy arrays
		star_dict['KIC'] = np.array(star_dict['KIC'])
		star_dict['RA'] = np.array(star_dict['RA'])
		star_dict['DEC'] = np.array(star_dict['DEC'])
		star_dict['Dnu'] = np.array(star_dict['Dnu'])
		star_dict['DPi1'] = np.array(star_dict['DPi1'])
		star_dict['spectra'] = np.array(star_dict['spectra'])
		return star_dict

if __name__ == '__main__':
	kepler_data = KeplerPeriodSpacing()
	kepler = kepler_data.get_data()
