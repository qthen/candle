# Spectra data embeddings

Spectra data is huge (10K datapoints) and many of it may not even be relevant to the classification task of red-clump stars, because of this, dimensionality reduction is a very useful approach at:

1. Preventing overfitting of the data since data set is ~3000 stars
2. Speed up computation time
3. Extract useful features

Here, I try to reduce the dimensionality of star spectra via. unsupervised and supervised learning. 

## Implementations

### SpectralEmbeddingPCA
Using unsupervised PCA to find a subspace for the spectra, typical reduction by a factor of 10 (~8.5K to 100)


## TO-DO

### Autoencoders
Semi-supervised auto encoding for the stellar spectra

### Few-shot learning
Learning embedding via supervised learning by learning a prototypical embedding for each class (RC, RGB, and potentially secondary RC). 