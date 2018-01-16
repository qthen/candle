# Class for generating 
from AstroData import AstroData
from astropy.table import Table
from astropy.table import Column
import numpy as np
import os.path

class KeplerPeriodSpacing(AstroData):

	'''
	Returns a numpy array of SPECTRA and their corresponding astroseismic parameters as labels
	'''
	def get_data(self):

		if not os.path.exists('../tables/KIC_A87/table1.dat'):
			# This gets me the PS and astroseismic parameters from the KIC, but not the RA or DE
			data_ps = Table.read("../tables/A87/table2.dat", format="ascii")

			# This gets me the RA and DE of the stars above into a dict
			data_ra_de = {}
			with open("../tables/KIC_A87/data_0.csv") as fp:
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

		table_kic = Table.read('../tables/KIC_A87/table1.dat', format='ascii')
		print(table_kic)

if __name__ == '__main__':
	kepler_data = KeplerPeriodSpacing()
	kepler = kepler_data.get_data()
