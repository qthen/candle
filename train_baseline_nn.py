from data.Kepler import KeplerPeriodSpacing
from models.spectra_embeddings.PCA import SpectralEmbeddingPCA
from models.regression.BaselineNN import BaselineNN
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

# Constants to use to train the BaselineNN
NUMBER_OF_STARS = 3000
EMBEDDING_DIMENSION = 100

# Optimizers
adam_opt = Adam(0.001)
sgd_opt = SGD(0.00001, momentum=0.9, decay=0.0, nesterov=True)

# Get the Kepler data with DPi1, Dnu and APOGEE spectra
kepler = KeplerPeriodSpacing()
data = kepler.get_data(max_number_of_stars = NUMBER_OF_STARS)

# Plotting the data of PS and large seperation - you can see a large cluster of 2 groups
ps = data['DPi1']
delta_v = data['Dnu']
plt.scatter(delta_v, ps)
plt.xlabel('Dnu')
plt.ylabel('DPi1')
plt.show()

# First reduce dimension to 10
pca = SpectralEmbeddingPCA(E_D = EMBEDDING_DIMENSION)

# Train on 90% and test on last 10%
pca.fit(data['spectra'][0:int(0.9*NUMBER_OF_STARS)])
spectra_data = pca.embed(data['spectra'])

# Model training and fitting
model = BaselineNN(S_D = EMBEDDING_DIMENSION)
model.compile(optimizer=adam_opt)
history = model.fit(spectra_data, [data['DPi1'], data['Dnu']], validation_split=0.1, epochs = 100, batch_size = 32)

# Plot the training data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training loss', 'Test loss'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the MAPE
plt.plot(history.history['DPi1_mean_absolute_percentage_error'])
plt.plot(history.history['Dnu_mean_absolute_percentage_error'])
plt.plot(history.history['val_DPi1_mean_absolute_percentage_error'])
plt.plot(history.history['val_Dnu_mean_absolute_percentage_error'])
plt.legend(['PS Training MAPE', 'Δv Training MAPE', 'PS Validation MAPE', 'Δv Validation MAPE'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('MAPE')
plt.show()

# model.compile(optimizer = sgd_opt)
# model.fit(spectra_data, data['DPi1'], epochs=1000, validation_split=0.1)



# # Plotting some random star spectra
# spectra = data['spectra'][0]
# plt.plot([i for i in range(0, len(spectra))], spectra)
# plt.show()

# pca = SpectralEmbeddingPCA(E_D = 2)
# pca.fit(np.array(data['spectra'][0:3000]))
# y = pca.embed(np.array(data['spectra'][3000:4000]))
# plt.scatter(y[:,0], y[:,1])
# plt.show()