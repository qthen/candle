from data.Kepler import KeplerPeriodSpacing
from models.spectra_embeddings.PCA import SpectralEmbeddingPCA
from models.regression.BaselineNN import BaselineNN
from keras.optimizers import SGD, Adam
from random import choice
import matplotlib.pyplot as plt
import numpy as np
import services.DataVisualization as DataVisualization

'''
General file for presenting findings and starting the model and data from scratch
'''

# Generate each version of the Kepler data
# Get the Kepler data with DPi1, Dnu and APOGEE spectra
kepler = KeplerPeriodSpacing()
# data_kepler_1 = kepler.get_data(version = 1, standardize = False)
data_kepler_2 = kepler.get_data(version = 2, standardize = False)

# Plot some PS as a function of Δv from version 2
plt.scatter(data_kepler_2['Dnu'], data_kepler_2['PS'], color="#F89406", alpha = 0.5)
plt.plot([0, 26], [100, 175], linestyle='--', color='#013243')
plt.xlim(xmin=0, xmax=26)
plt.ylim(ymin=0, ymax=400)
plt.title("Period spacing in red giants")
plt.xlabel("Δv - Large frequency separation")
plt.ylabel("Period Spacing")
plt.show()

# Plot the effective temperature as a function of log(g)
plt.scatter(data_kepler_2['T_eff'], data_kepler_2['logg'], color="#F89406", alpha = 0.5)
plt.title("Effective temperature and log(g) distribution")
plt.xlabel("T_eff")
plt.ylabel("log(g)")


# Plot reconstruction of spectra from PCA, this is actualy a surprising find, but it is not obvious from the graph. All spectra are fairly simiilar and PCA captures the overall structure, but the problem is that visually, there doesn't seem to be a huge improvement when plotting 1 component PCA (ridiculous) vs. 5 components, but there are idosyncranics in the data
# The large difference is clearly shown, however, when showing the distance between the reconstructed spectra 

N = len(data_kepler_2['KIC'])

# PCA fitting with 1 components
pca_1 = SpectralEmbeddingPCA(E_D = 1)
pca_1.fit(data_kepler_2['spectra'][0:int(0.9*N)])

# PCA fitting with 5 components
pca_5 = SpectralEmbeddingPCA(E_D = 5)
pca_5.fit(data_kepler_2['spectra'][0:int(0.9*N)])

# PCA fitting with 50 components
pca_50 = SpectralEmbeddingPCA(E_D = 50)
pca_50.fit(data_kepler_2['spectra'][0:int(0.9*N)])

random_spectra_from_validation = choice(data_kepler_2['spectra'][int(0.9*N):])

reduced_spectra_1 = pca_1.embed(np.array([random_spectra_from_validation]))
reconstructed_spectra_1 = pca_1.PCA.inverse_transform(reduced_spectra_1)[0]

reduced_spectra_5 = pca_5.embed(np.array([random_spectra_from_validation]))
reconstructed_spectra_5 = pca_5.PCA.inverse_transform(reduced_spectra_5)[0]

reduced_spectra_50 = pca_50.embed(np.array([random_spectra_from_validation]))
reconstructed_spectra_50 = pca_50.PCA.inverse_transform(reduced_spectra_50)[0]

DataVisualization.plot_spectras([random_spectra_from_validation, reconstructed_spectra_1, reconstructed_spectra_5], labels = ['Original Spectra', 'n = 1: Reconstructed Spectra', 'n = 5: Reconstructed spectra'])

dist_1 = np.linalg.norm(random_spectra_from_validation - reconstructed_spectra_1)
dist_2 = np.linalg.norm(random_spectra_from_validation - reconstructed_spectra_5)
dist_3 = np.linalg.norm(random_spectra_from_validation - reconstructed_spectra_50)

print("n = 1: {}, n = 5: {}, n = 50: {}".format(dist_1, dist_2, dist_3))

# Plot the distances from the reconstructed spectra on a bar graph to demonstrate the hard to notice difference
bar1 = plt.bar(1, dist_1, alpha = 0.9)
bar2 = plt.bar(2, dist_2, alpha = 0.9)
bar3 = plt.bar(3, dist_3, alpha = 0.9)

plt.xticks([1, 2, 3], ['n = 1', 'n = 5', 'n = 50'])
plt.title('Euclidean distance from reconstructed spectra to original spectra from n PCA components')
plt.tight_layout()
plt.show()

# # Plot the decreasing distance as a function of n components PCA
# x = []
# y = []
# for i in range(1, 100):
# 	pca = SpectralEmbeddingPCA(E_D = i)
# 	pca.fit(data_kepler_2['spectra'][0:int(0.9*N)])
# 	reduced_spectra = pca.embed(np.array([random_spectra_from_validation]))
# 	reconstructed_spectra = pca.PCA.inverse_transform(reduced_spectra)[0]
# 	x.append(i+1)
# 	y.append(np.linalg.norm(random_spectra_from_validation - reconstructed_spectra))

# plt.plot(x, y)
# plt.tight_layout()
# plt.xlabel('Number of PCA components')
# plt.ylabel('Euclidean distance')
# plt.title("Reconstruc")
# plt.show()
