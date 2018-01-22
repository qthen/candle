'''
Training entry point for the baseline convolutional neural network
'''
from data.Kepler import KeplerPeriodSpacing
from models.spectra_embeddings.PCA import SpectralEmbeddingPCA
from models.regression.BaselineCNN import BaselineCNN as BaselineCNNRegression
from keras.optimizers import SGD, Adam
import argparse
import numpy as np

# Argument line parsing first
parser = argparse.ArgumentParser(description='Train baseline convolutional neural network')
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs to train for')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size to use')
parser.add_argument('--optimizer', dest='optimizer', type=str, default="adam", help='Optimizer to use')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='Learning rate to use')
parser.add_argument('--regression', dest='regression', type=bool, default=True, help="Regression task?")
args = parser.parse_args()
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size

optimizer = SGD(args.lr, momentum=0.9, decay=0.0, nesterov=True)
if args.optimizer == 'adam':
	optimizer = Adam(args.lr)

# Get the Kepler data with DPi1, Dnu and APOGEE spectra
kepler = KeplerPeriodSpacing()
data = kepler.get_data(version = 2, standardize = False, use_steps = True)
N = len(data['KIC'])

# Model training and fitting
if args.regression:
	# Regression task
	model = BaselineCNNRegression(S_D = data['spectra'].shape[1])
	model.compile(optimizer=optimizer)
	history = model.fit(data['spectra'], [data['PS'], data['Dnu']], validation_split=0.1, epochs = epochs, batch_size = batch_size)