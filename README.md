# Candle
### Spectral discrimination of helium burning stars among red giants

`Candle` is an in-progress overview of machine learning techniques to classify red clump stars among arbitrary APOGEE stars, primarily composing of red giants as of DR14. 

When Kepler missions observed star brightness oscillations for exoplanet detection, we also discovered a rich separation of red clumps and red giant branch stars in asteroseismic space. As this data is expensive to measure and not as widely available as spectral data on stars, it is of interest to see spectral applications to discerning red clump stars, which are standard candles due to the recent growing availability and size of APOGEE sky surveys and their resolution. 

The current model is based on fine-tuned parameters and best results from exploration of a variety of techniques and models that yield not only good classification on Kepler measured red giants, but also good generalization to an arbitary distribution of stars from APOGEE DR14. The current model gets around a 10-fold cross validation 98.58% accuracy on the Kepler dataset (import from `data.KeplerPeriodSpacing`) and with results that agree with theoretical understandings of the red giant branch. 

For CSC494 research project under the supervison of Professor Bovy - <a href="http://astro.utoronto.ca/~bovy/">Jo Bovy</a>

## Project description

> Red clump stars are helium burning red giants, late in their stellar evolution and are of significant importance in astronomy. They serve as a standard candle, cosmological objects with known absolute magnitude, bright enough to be observed throughout the Milky Way, that fit the candidate for ideal distance measurements, one of the most important metrics for astronomers. Late in their age as a red-giant branch, these red-clump stars need to be distinguished from their closely related, still maturing hydrogen-burning red giants that have just finished hydrogen-core burning on the main sequence. This important distinguishment and recently, fastly growing availability and size of astronomy data has led to multiple papers and interests on techniques for discovering them from traditional methods through investigating individual stellar properities to applications from neural networks both with a high degree of success. In this project, we will utilize the APOGEE dataset that provides a comprehensive set of stellar spectra and parameters, and the Kepler observations of red-clump stars to investigate applications of traditional and non-traditional machine learning techniques to distinguish red clump stars, with the goals of beating the baseline and recent simple machine learning techniques that have had ~95% and 98% accuracy respectively. 

## Usage from saved spectra

To classify arbitary APOGEE stars from DR14 (7514 spectra pixel), we do the following

```
from models.classification.Candle import Candle # Import model
import pickle

# Pick out 3000 random APOGEE staars
pickle_out = open("pickles/APOGEE/apogee_sample_3000_stars.pickle", 'rb')
apogee_data = pickle.load(pickle_out)
pickle_out.close()

# Create candle model
candle_model = Candle()

# Get probabilities of RC according to Candle
probabilities = candle_model.predict(apogee_data['spectra'])
```

## Usage from sampling spectra

To sample random APOGEE spectra, use the data factory `data.APOGEE`

```
from data.APOGEE import APOGEE
from models.classification.Candle import Candle # Import model

# Get 100 random APOGEE stars
apogee_factory = APOGEE()
apogee_data = apogee_factory.grab_sample(N=100)

# Create candle model
candle_model = Candle()

# Get probabilities of RC according to Candle
probabilities = candle_model.predict(apogee_data['spectra'])
```

## Visualizing classifications

To visualize our classifications, one of the most common methods is to view it in Teff-log(g) space that roughly corresponds to the HR diagram of stars. We can use the `services.ModelPerformanceVisualization` service to view it

```
from models.classification.Candle import Candle
from services.ModelPerformanceVisualization import plot_classifications
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Pick out 3000 APOGEE staars
pickle_out = open("pickles/APOGEE/apogee_sample_3000_stars.pickle", 'rb')
apogee_data = pickle.load(pickle_out)
pickle_out.close()

# Create candle model
candle_model = Candle()

# Classify them according to Candle
probabilities = candle_model.predict(apogee_data['spectra'])
plot_classifications(apogee_data['Teff'], apogee_data['logg'], probabilities)
plt.show()
```

This will return a visualization like
![Classifications](/plots/readme/fig1.png)
