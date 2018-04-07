# A collection of functions to visualize a model's performance
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

'''
Plots the Δv - Π_1 space given the status of each red giant
Input:
	dnu - DNU parameters
	ps - PS parameters
	status - The status of each red giant, 0 for rgb, >= 1 for RC
'''
def plot_dnu_ps_space(dnu, ps, status, title="Δv - Π_1 space"):
	N = len(dnu)
	rgb_indices = [i for i in range(0, N) if status[i] == 0]
	rc_indices = [i for i in range(0, N) if status[i] >= 1]
	ax_rgb = plt.scatter(dnu[rgb_indices], ps[rgb_indices],  color="#F03434", alpha=0.5, marker='v')
	ax_rc = plt.scatter(dnu[rc_indices], ps[rc_indices],  color="#F89406", alpha=0.5)
	plt.legend((ax_rgb, ax_rc), ("RGB", "RC"))
	plt.title(title)
	plt.xlabel("Δv - Large frequency separation")
	plt.ylabel("Π_1 - Period spacing")


'''
Plots useful plots of a difficult star in respect to many spaces

Current process is to take a given stellar spectra
Truth and regression space expects the following spaces in order (as a list)
	Δv, Π_1, log(g), T_eff, status

These spaces are assumed to be non-normalized
'''
def plot_difficult_star(kic, target_idx, spectral_space, truth_space, regression_space):
	N = len(spectral_space)
	truth_dnu, truth_ps, truth_logg, truth_teff, truth_rc = truth_space
	reg_dnu, reg_ps, reg_logg, reg_teff, reg_rc = regression_space

	truth_rgb_indices = [i for i in range(0, N) if truth_rc[i] == 0]
	truth_rc_indices = [i for i in range(0, N) if truth_rc[i] >= 1]

	target_spectra = spectral_space[target_idx]

	# Find the closest star to this star in spectral space and plot it on the log(g)-Teff diagram
	spectral_closest_idx = k_closest_vectors(spectral_space, target_spectra, k = 10)
	spectral_closest_idx = [idx for idx in spectral_closest_idx if idx != target_idx]
	spectral_closest_rc_idx = [idx for idx in spectral_closest_idx if truth_rc[idx] >= 1]
	spectral_closest_rgb_idx = [idx for idx in spectral_closest_idx if truth_rc[idx] == 0]

	# Plot the spectra space in the background, faded
	plt.scatter(truth_teff[truth_rgb_indices], truth_logg[truth_rgb_indices], color="#F03434", alpha=0.05, marker='v')
	plt.scatter(truth_teff[truth_rc_indices], truth_logg[truth_rc_indices], color="#F89406", alpha=0.05)
	rgb_ax = plt.scatter(truth_teff[spectral_closest_rgb_idx], truth_logg[spectral_closest_rgb_idx], color="#F03434", alpha=1, marker='v')
	rc_ax = plt.scatter(truth_teff[spectral_closest_rc_idx], truth_logg[spectral_closest_rc_idx], color="#F89406", alpha=1)
	target_ax = plt.scatter(truth_teff[target_idx], truth_logg[target_idx], color="#3498DB", marker='s')

	# Annotate distances from each star
	for idx in spectral_closest_idx:
		plt.annotate("{:.2f}".format(np.linalg.norm(spectral_space[idx] - spectral_space[target_idx])), (truth_teff[idx], truth_logg[idx]))
	plt.xlim(5200, 4500)
	plt.ylim(3.5, 2.25)
	plt.xlabel("Effective temperature - T_eff")
	plt.ylabel("Surface gravity - log(g)")
	plt.title("KIC {} closest giants in spectral space".format(kic))
	plt.legend((rgb_ax, rc_ax, target_ax), ("RGB", "RC", "KIC {}".format(kic)))
	plt.show()

	# Find the closest stars in regression space based on regression parameters and then based on truth parameters
	dnu_ps_reg_space = np.column_stack([scale(reg_dnu), scale(reg_ps)])
	dnu_ps_truth_space = np.column_stack([scale(truth_dnu), scale(truth_ps)])
	closest_idx_reg_params = k_closest_vectors(dnu_ps_reg_space, dnu_ps_reg_space[target_idx], k = 5)
	closest_idx_truth_params = k_closest_vectors(dnu_ps_reg_space, dnu_ps_truth_space[target_idx], k = 5)
	closest_idx_reg_params = [idx for idx in closest_idx_reg_params if idx != target_idx]
	closest_idx_truth_params = [idx for idx in closest_idx_truth_params if idx != target_idx]

	# Show where the closest stars are
	plt.scatter(reg_dnu[truth_rgb_indices], reg_ps[truth_rgb_indices], color="#F03434", alpha=0.05, marker='v')
	plt.scatter(reg_dnu[truth_rc_indices], reg_ps[truth_rc_indices], color="#F89406", alpha=0.05)
	reg_ax = plt.scatter(reg_dnu[closest_idx_reg_params], reg_ps[closest_idx_reg_params], color="#F89406")
	truth_ax = plt.scatter(reg_dnu[closest_idx_truth_params], reg_ps[closest_idx_truth_params], color="#F03434")
	target_ax = plt.scatter(reg_dnu[target_idx], reg_ps[target_idx], color="#3498DB", marker='s')
	target_true_ax = plt.scatter(truth_dnu[target_idx], truth_ps[target_idx], color="#3498DB", marker='p')
	plt.title("Where the closest stars came from in Δv - Π_1 regression space")
	plt.xlabel("Δv - large frequency separation")
	plt.ylabel("Π_1 - Period spacing")
	plt.legend((truth_ax, reg_ax, target_ax), ("Closest giants to true params", "Closest giants to regressed params", "Regressed KIC {}".format(kic), "True KIC {}".format(kic)))
	plt.show()



'''
Returns k closest vectors to some target vector given a list of points in R^n
Input:
	space - The list of points in R^n
	target - The target vector in R^n
	k - k closest vectors to return
Output: indices of closest vectors
'''
def k_closest_vectors(space, target, k = 5):
	d = np.linalg.norm(space - target, axis = 1)
	return d.argsort()[0:k]

'''
Plots performance of binary classification into logg-Teff space
Correctly labelled stars are faint (alpha = 0.1) while incorrect labelled stars are opaque
Input:
	logg - x-axis
	teff - y-axis
	ground_truth - Ground truth for labels
	probabilities - Predicted probabilities of labels
	title - The title of the scatter plot
	teff_max - The max temperature
	teff_min - The min temperature
	logg_max - Max surface gravity
	logg_min - Min surface gravity
'''
def plot_binary_classification(logg, teff, ground_truth, probabilities, title="Binary Classification", teff_max = 5200, teff_min = 4500, logg_max = 3.5, logg_min = 2.25):
	rc_stars = np.array([[teff[i], logg[i]] for i in range(0, min(len(logg), len(teff))) if ground_truth[i] == 1 and probabilities[i] >= 0.5])
	rgb_stars = np.array([[teff[i], logg[i]] for i in range(0, min(len(logg), len(teff))) if ground_truth[i] == 0 and probabilities[i] < 0.5])
	rc_stars_misclassified = np.array([[teff[i], logg[i]] for i in range(0, min(len(logg), len(teff))) if ground_truth[i] == 1 and probabilities[i] < 0.5])
	rgb_stars_misclassified = np.array([[teff[i], logg[i]] for i in range(0, min(len(logg), len(teff))) if ground_truth[i] == 0 and probabilities[i] >= 0.5])
	plt.title(title)
	plt.xlim(teff_max, teff_min)
	plt.ylim(logg_max, logg_min)
	rc_ax = plt.scatter(rc_stars[:,0], rc_stars[:,1], color="#F89406", alpha=0.05)
	rgb_ax = plt.scatter(rgb_stars[:,0], rgb_stars[:,1], color="#F03434", alpha=0.05)
	rgb_ax_mis = plt.scatter(rgb_stars_misclassified[:,0], rgb_stars_misclassified[:,1], color="#F03434", alpha=1, marker='v')
	if len(rc_stars_misclassified) == 0:
		rc_ax_mis = plt.scatter([], [], color="#F89406", alpha=1, marker='^')
	else:
		rc_ax_mis = plt.scatter(rc_stars_misclassified[:,0], rc_stars_misclassified[:,1], color="#F89406", alpha=1, marker='^')
	plt.legend((rc_ax, rgb_ax, rgb_ax_mis, rc_ax_mis), ('RC stars classified correctly', 'RGB stars classified correctly', 'RGB stars mislassified', 'RC stars misclassified'))

'''
Plot classifications

Given teffs, loggs and an array of probabilities and a threshold, plots the classifications there the probability is p(x = red clump)
Inputs:
	teff - (n, ) ndarray of temperature
	logg - (n, ) ndarray of logg
	probabilities - (n, ) ndarray of probabilities
'''
def plot_classifications(teff, logg, probabilities, threshold = 0.5, title="Model classification on APOGEE stars"):
	N = len(probabilities)
	rc_idx = [i for i in range(0, N) if probabilities[i] >= threshold]
	rgb_idx = [i for i in range(0, N) if probabilities[i] < threshold]
	rgb_ax = plt.scatter(teff[rgb_idx], logg[rgb_idx], marker='v', color="#F03434", alpha=0.3)
	rc_ax = plt.scatter(teff[rc_idx], logg[rc_idx], color="#F89406", alpha=0.3)
	plt.legend((rc_ax, rgb_ax), ('RGB', 'RC'))
	plt.xlim(5400, 3500)
	plt.ylim(4.5, 0)
	plt.title(title)


'''
Plots the predicted PS with the truth PS
Input:
	y_ps - [Float], ndarray of PS
	y_pred_ps - [Float], ndarray of predicted PS
'''
def plot_ps_vs_pred_ps(y_ps, y_pred_ps):
	plt.scatter(y_ps, y_pred_ps, alpha = 0.5)
	plt.xlabel("Vrad 2016, PS")
	plt.ylabel("Predicted PS")
	plt.title("Predicted period spacing from spectra")
	plt.xlim(xmin =0, xmax=360)
	plt.ylim(ymin =0, ymax=360)
	plt.plot([0, 360], [0, 360], color='#34495e', linestyle='-', linewidth=1)
	plt.show()

'''
Plots the predicted Dnu with the truth Dnu
Inputs:
	y_Dnu - [Float], ndarray of Δv
	y_pred_Dnu - [Float], ndarray of Δv
'''
def plot_dnu_vs_pred_dnu(y_Dnu, y_pred_Dnu):
	plt.scatter(y_Dnu, y_pred_Dnu, alpha = 0.5)
	plt.xlabel("Vrad 2016, Δv")
	plt.ylabel("Predicted Δv")
	plt.title("Predicted Δv from spectra")
	plt.plot([0, 20], [0, 20], color='#34495e', linestyle='-', linewidth=1)
	plt.xlim(xmin =0, xmax=20)
	plt.ylim(ymin =0, ymax=20)
	plt.show()

'''
Plots classification of stars with PS as a function of Dnu
Input:
	Dnu - Array of Dnu for the stars (x-axis)
	PS - Array of PS of the stars (y-axis)
	classifications - Classifications as an array
	class_labels - Array of labels
	title - Title of the classification graph
'''
def plot_classification(Dnu, PS, classifications, class_labels, title = "Classification of Kepler giants"):
	K = len(class_labels) # Number of different classes, classifications should have K distinct integers
	y_classes = [[] for i in range(0, K)]
	for i in range(0, len(classifications)):
		y_classes[classifications[i]].append([Dnu[i], PS[i]])
	ax = plt.subplot(111)
	categories = []
	for k in range(0, K):
		data = np.array(y_classes[k])
		categories.append(plt.scatter(data[:,0], data[:,1],  alpha=0.5))
	plt.xlim(xmin=0, xmax=20)
	plt.ylim(ymin=0, ymax=400)
	plt.xlabel("Δv - large frequency separation")
	plt.ylabel("Period spacing")
	
	plt.legend(categories, class_labels)
	plt.title(title)
	plt.show()

'''
Plots the given stellar predictions on subplots
'''
def plot_all(y_ps, y_pred_ps, y_Dnu, y_pred_Dnu, y_Teff, y_pred_Teff, y_logg, y_pred_logg):
	plt.subplot(221)
	plt.scatter(y_ps, y_pred_ps, alpha = 0.6, color="#F03434")
	plt.xlabel("Vrad 2016, PS")
	plt.ylabel("Predicted PS")
	plt.title("Predicted period spacing from spectra")
	plt.xlim(xmin =0, xmax=380)
	plt.ylim(ymin =0, ymax=380)
	plt.plot([0, 380], [0, 380], color='#34495e', linestyle='-', linewidth=1)

	plt.subplot(222)
	plt.scatter(y_Dnu, y_pred_Dnu, alpha = 0.6)
	plt.xlabel("Vrad 2016, Δv")
	plt.ylabel("Predicted Δv")
	plt.title("Predicted Δv from spectra")
	plt.plot([0, 20], [0, 20], color='#34495e', linestyle='-', linewidth=1)
	plt.xlim(xmin =0, xmax=20)
	plt.ylim(ymin =0, ymax=20)

	plt.subplot(223)
	plt.scatter(y_Teff, y_pred_Teff, alpha = 0.6, color="#F89406")
	plt.xlabel("Vrad 2016, T_eff")
	plt.ylabel("Predicted T_eff")
	plt.title("Predicted T_eff from spectra")
	plt.plot([4400, 5200], [4400, 5200], color='#34495e', linestyle='-', linewidth=1)
	plt.xlim(xmin =4400, xmax=5200)
	plt.ylim(ymin =4400, ymax=5200)

	plt.subplot(224)
	plt.scatter(y_logg, y_pred_logg, alpha = 0.6, color="#6C7A89")
	plt.xlabel("Vrad 2016, log(g)")
	plt.ylabel("Predicted log(g)")
	plt.title("Predicted log(g) from spectra")
	plt.plot([2, 3.5], [2, 3.5], color='#34495e', linestyle='-', linewidth=1)
	plt.xlim(xmin =2, xmax=3.5)
	plt.ylim(ymin =2, ymax=3.5)

	plt.tight_layout()
	plt.show()
