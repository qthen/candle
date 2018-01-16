from data.Kepler import KeplerPeriodSpacing # Kepler data
import matplotlib.pyplot as plt

kepler = KeplerPeriodSpacing()
data = kepler.get_data(max_number_of_stars = 100)


# Plotting the data of PS and large seperation - you can see a large cluster of 2 groups
ps = data['DPi1']
delta_v = data['Dnu']
plt.scatter(delta_v, ps)
plt.xlabel('Dnu')
plt.ylabel('DPi1')
plt.show()

# Plotting some random star spectra
spectra = data['spectra'][0]
plt.plot([i for i in range(0, len(spectra))], spectra)
plt.show()


