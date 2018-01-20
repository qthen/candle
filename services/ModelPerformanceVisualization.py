# A collection of functions to visualize a model's performance
import matplotlib.pyplot as plt

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
Plots the given stellar predictions on subplots
'''
def plot_all(y_ps, y_pred_ps, y_Dnu, y_pred_Dnu):
	plt.subplot(121)
	plt.scatter(y_ps, y_pred_ps, alpha = 0.6, color="#F03434")
	plt.xlabel("Vrad 2016, PS")
	plt.ylabel("Predicted PS")
	plt.title("Predicted period spacing from spectra")
	plt.xlim(xmin =0, xmax=360)
	plt.ylim(ymin =0, ymax=360)
	plt.plot([0, 360], [0, 360], color='#34495e', linestyle='-', linewidth=1)
	plt.subplot(122)
	plt.scatter(y_Dnu, y_pred_Dnu, alpha = 0.6)
	plt.xlabel("Vrad 2016, Δv")
	plt.ylabel("Predicted Δv")
	plt.title("Predicted Δv from spectra")
	plt.plot([0, 20], [0, 20], color='#34495e', linestyle='-', linewidth=1)
	plt.xlim(xmin =0, xmax=20)
	plt.ylim(ymin =0, ymax=20)
	plt.tight_layout()
	plt.show()
