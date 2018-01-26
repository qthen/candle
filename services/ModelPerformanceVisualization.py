# A collection of functions to visualize a model's performance
import matplotlib.pyplot as plt
import numpy as np

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
		plt.title(title)
		plt.legend(categories, class_labels)
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
