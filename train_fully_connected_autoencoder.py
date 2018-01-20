from data.Kepler import KeplerPeriodSpacing
from models.spectra_embeddings.Autoencoder import FullyConnectedAutoencoder, ConvolutionalAutoencoder

# Get the Kepler data with DPi1, Dnu and APOGEE spectra
kepler = KeplerPeriodSpacing()
data = kepler.get_data(version = 2, standardize = False, use_steps = True)
N = len(data['KIC'])

encoder = FullyConnectedAutoencoder(data['spectra'].shape[1], 50)
encoder_conv = ConvolutionalAutoencoder(data['spectra'].shape[1], 50)
encoder_conv.fit(data['spectra'])