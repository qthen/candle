from data.Kepler import KeplerPeriodSpacing
from models.regression.BaselineCNN import BaselineCNN
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

# Constants to use to train the BaselineNN
NUMBER_OF_STARS = 200

# Optimizers
adam_opt = Adam(0.01)
sgd_opt = SGD(0.0001, momentum=0.9, decay=0.0, nesterov=True)

# Get the Kepler data with DPi1, Dnu and APOGEE spectra
kepler = KeplerPeriodSpacing()
data = kepler.get_data(max_number_of_stars = NUMBER_OF_STARS, use_steps = True)

# Plotting the data of PS and large seperation - you can see a large cluster of 2 groups
ps = data['DPi1']
delta_v = data['Dnu']
plt.scatter(delta_v, ps)
plt.xlabel('Dnu')
plt.ylabel('DPi1')
plt.show()

# Model training and fitting
model = BaselineCNN(S_D = data['spectra'].shape[1])
model.compile(optimizer=sgd_opt)
history = model.fit(data['spectra'], [data['DPi1'], data['Dnu']], validation_split=0.1, epochs = 100, batch_size = 32)

# # Plot the training data
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['Training loss', 'Test loss'], loc='upper left')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# # Plot the MAPE
# plt.plot(history.history['DPi1_mean_absolute_percentage_error'])
# plt.plot(history.history['Dnu_mean_absolute_percentage_error'])
# plt.plot(history.history['val_DPi1_mean_absolute_percentage_error'])
# plt.plot(history.history['val_Dnu_mean_absolute_percentage_error'])
# plt.legend(['PS Training MAPE', 'Δv Training MAPE', 'PS Validation MAPE', 'Δv Validation MAPE'], loc='upper left')
# plt.xlabel('Epoch')
# plt.ylabel('MAPE')
# plt.show()

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