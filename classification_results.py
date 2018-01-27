from data.Kepler import KeplerPeriodSpacing
from models.spectra_embeddings.PCA import SpectralEmbeddingPCA
from models.classification.GaussianNaiveBayes import GaussianNaiveBayes
from models.classification.LogisticRegression import LogisticRegression
from models.classification.KNearestNeighbors import KNearestNeighbors
from models.classification.BaselineNN import BaselineNN as BaselineNNBinaryClassification
import argparse
import numpy as np

# Argument line parsing first
parser = argparse.ArgumentParser(description='Logistic regression')
parser.add_argument('--components', dest='components', type=int, default=100, help='Number of PCA components')
args = parser.parse_args()
components = args.components

# Get the Kepler data with DPi1, Dnu and APOGEE spectra
kepler = KeplerPeriodSpacing()
data = kepler.get_data(version = 2, standardize = False)
N = len(data['KIC'])

# First reduce dimension to components
pca = SpectralEmbeddingPCA(E_D = components)

# Train on 90% and test on last 10%
pca.fit(data['spectra'][0:int(0.9*N)])
spectra_data = pca.embed(data['spectra'])

training_data, training_labels = spectra_data[0:int(0.9*N)], data['RC'][:int(0.9*N)]
validation_data, validation_labels = spectra_data[int(0.9*N):], data['RC'][int(0.9*N):]
validation_plots = [data['PS'][int(0.9*N):], data['Dnu'][int(0.9*N):]]

# Baseline NN
model = BaselineNNBinaryClassification(S_D = components)
model.compile(optimizer='adam', metrics=['acc'])
model.fit(training_data, [training_labels], validation_split=0.0, epochs = 7, batch_size = 32)
model.score(validation_data, validation_labels)
model.judge(validation_data, validation_plots)

# Gaussian Model
model = GaussianNaiveBayes()
model.compile()
model.fit(training_data, training_labels)
model.judge(validation_data, validation_plots)
model.score(validation_data, validation_labels)

# K-NN Model
model = KNearestNeighbors()
model.compile()
model.fit(training_data, training_labels)
model.judge(validation_data, validation_plots)
model.score(validation_data, validation_labels)

# Logistic regression model
model = LogisticRegression()
model.compile(max_iterations = 100)
model.fit(training_data, training_labels)
model.judge(validation_data, validation_plots)
model.score(validation_data, validation_labels)