# candle
### Spectral discrimination of helium burning stars among red giants

candle is an in-progress overview of machine learning techniques to distinguish red clump stars from red giant branch stars while only relying on spectral data from APOGEE, with the goals of beating Cannon (2% contamination rate) with fast performance.

As asteroseismic data is not as widely available as spectral data on stars and with the recent growing availability and size of APOGEE sky surveys and their resolution, it is of interest to see spectral applications to discerning red clump stars, which are standard candles, from their younger red giant branch stars. 

For CSC494 research project under the supervison of Professor Bovy - <a href="http://astro.utoronto.ca/~bovy/">Jo Bovy</a>

## Project description

> Red clump stars are helium burning red giants, late in their stellar evolution and are of significant importance in astronomy. They serve as a standard candle, cosmological objects with known absolute magnitude, bright enough to be observed throughout the Milky Way, that fit the candidate for ideal distance measurements, one of the most important metrics for astronomers. Late in their age as a red-giant branch, these red-clump stars need to be distinguished from their closely related, still maturing hydrogen-burning red giants that have just finished hydrogen-core burning on the main sequence. This important distinguishment and recently, fastly growing availability and size of astronomy data has led to multiple papers and interests on techniques for discovering them from traditional methods through investigating individual stellar properities to applications from neural networks both with a high degree of success. In this project, we will utilize the APOGEE dataset that provides a comprehensive set of stellar spectra and parameters, and the Kepler observations of red-clump stars to investigate applications of traditional and non-traditional machine learning techniques to distinguish red clump stars, with the goals of beating the baseline and recent simple machine learning techniques that have had ~95% and 98% accuracy respectively. 


## Classification task
Based on a line of best fit with data from `Vrad, 2016` that is able to predict red-clumps with high accuracy, ground truth in red clump status is define to be a point lying above a hyperplane when red giant's period spacing are plotted as a function of Δv, in this case, the best fit line was:

<p style="text-align:center;">
y = 75/26x + 100
</p>