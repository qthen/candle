# Regression models

Regression models for stellar parameters on stars that are inferred from stellar spectra. These stellar parameters are often ones we have interest in or should be heavily relevant in predicting helium-core burning in stars. 

Anecdotally, the power a model is able to predict stellar features like effective temperautre, log(g), period spacing, etc. might be a good indicator at how well it is able to handle the classification task ("Is a star burning helium in its core?") as these features have been shown to relate differently in red clump stars and red giant branch stars. 

## Model implementations

The current model implementations, most take in a dimensionally reduced vector of the star spectra except for convolutional networks

### BaselineNN
A baseline neural network regression that has 2 hidden layers with 128 and 64 neurons respectively for around 8,000 learnable parameters. Mainly using it for evaluating how well an embedding of stellar spectra is. Preliminary results with 100 PCA components has the following regression results after 100 training iterations:

	loss                                    : 1611.0846
	DPi1_loss                               : 1610.5841
	Dnu_loss                                : 0.5005
	DPi1_mean_absolute_error                : 28.6636
	DPi1_mean_absolute_percentage_error     : 17.2518
	Dnu_mean_absolute_error                 : 0.4684
	Dnu_mean_absolute_percentage_error      : 7.1293
	val_loss                                : 1818.0663
	val_DPi1_loss                           : 1817.6984
	val_Dnu_loss                            : 0.3680
	val_DPi1_mean_absolute_error            : 27.0088
	val_DPi1_mean_absolute_percentage_error : 14.0153
	val_Dnu_mean_absolute_error             : 0.3383
	val_Dnu_mean_absolute_percentage_error  : 5.6662

And around a 98% accuracy on classification on the validation set. 

### BaselineCNN
A baseline convolutional network that has 3 convolutional layers following by MaxPool, then connected to two dense layers. The convolutional layers are responsible for downsizing the stellar spectra to a ~100 size vector before handing it to the fully connected layers. Currently, this model, which should intuitively outperform PCA -> BaselineNN actually performs horribly. A few thoughts I have may be because I've been trying to keep the number of convolutional learnable parameters relatively small, which is hard since we are using a large vector dimension (~8.5K) with a small sample size (~2K). In keeping it too small, the convolutional layer is too shallow or simple to learn anything useful, need to work on this.