'''
Training entry point for the baseline neural network, mainly used for evaluating embeddings on spectra
'''
from data.Kepler import KeplerPeriodSpacing
from models.spectra_embeddings.PCA import SpectralEmbeddingPCA
from models.regression.BaselineNN import BaselineNN as BaselineNNRegression
from models.classification.BaselineNN import BaselineNN as BaselineNNBinaryClassification
from keras.optimizers import SGD, Adam
import argparse
import numpy as np

# Argument line parsing first
parser = argparse.ArgumentParser(description='Train baseline neural network')
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs to train for')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size to use')
parser.add_argument('--optimizer', dest='optimizer', type=str, default="adam", help='Optimizer to use')
parser.add_argument('--components', dest='components', type=int, default=100, help='Number of PCA components')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='Learning rate to use')
parser.add_argument('--regression', dest='regression', type=bool, default=False, help="Regression task?")
parser.add_argument('--shared', dest='shared', type=bool, default=False, help="Share dense layers after embedding?")
args = parser.parse_args()
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
components = args.components

optimizer = SGD(args.lr, momentum=0.9, decay=0.0, nesterov=True)
if args.optimizer == 'adam':
	optimizer = Adam(args.lr)

# Get the Kepler data with DPi1, Dnu and APOGEE spectra
kepler = KeplerPeriodSpacing()
data = kepler.get_data(version = 2, standardize = False)
N = len(data['KIC'])

# First reduce dimension to 10
pca = SpectralEmbeddingPCA(E_D = components)

# Train on 90% and test on last 10%
pca.fit(data['spectra'][0:int(0.9*N)])
spectra_data = pca.embed(data['spectra'])

# Model training and fitting
if args.regression:
	if args.shared:
		# Regression task
		model = BaselineNNRegression(S_D = components, shared=True)
		model.compile(optimizer=optimizer, metrics=['mse'])
		history = model.fit(spectra_data, [np.column_stack(data['PS'], data['Dnu'])], validation_split=0.1, epochs = epochs, batch_size = batch_size)
		# Show model visualizations
		y_pred = model.judge(spectra_data[int(0.9*N):], [data['PS'][int(0.9*N):], data['Dnu'][int(0.9*N):]])
	else:
		# Regression task
		model = BaselineNNRegression(S_D = components, shared=False)
		model.compile(optimizer=optimizer)
		history = model.fit(spectra_data, [data['PS'], data['Dnu'],  data['T_eff'], data['logg']], validation_split=0.1, epochs = epochs, batch_size = batch_size)
		# Show model visualizations
		y_pred = model.judge(spectra_data[int(0.9*N):], [data['PS'][int(0.9*N):], data['Dnu'][int(0.9*N):], data['T_eff'][int(0.9*N):], data['logg'][int(0.9*N):]])
		model.save()

else:
	# Classification task
	model = BaselineNNBinaryClassification(S_D = components)
	model.compile(optimizer=optimizer, metrics=['acc'])
	history = model.fit(spectra_data, [data['RC']], validation_split=0.1, epochs = epochs, batch_size = batch_size)

	# Show model visualizations
	y_pred = model.judge((spectra_data[int(0.9*N):], data['KIC'][int(0.9*N):]), [data['PS'][int(0.9*N):], data['Dnu'][int(0.9*N):], data['RC'][int(0.9*N):]])

	model.save()
