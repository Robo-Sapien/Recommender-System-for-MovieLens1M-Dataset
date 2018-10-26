import numpy as np
from data_parser import *
from scipy import spatial
from array import *

from prediction import *

def RMSE(predited_rating, actual_rating):
	N = len(actual_rating)
	predited_rating = np.asarray(predited_rating)
	actual_rating = np.asarray(actual_rating)

	squared_deviation_mean = np.power(np.subtract(predited_rating,actual_rating),2)*1.0/N
	RMSEerror = np.power(np.sum(squared_deviation_mean),0.5)

	return RMSEerror

def Spearman_correlation(predited_rating, actual_rating):
	N = len(actual_rating)
	predited_rating = np.asarray(predited_rating)
	actual_rating = np.asarray(actual_rating)
	squared_deviation = np.power(np.subtract(predited_rating,actual_rating),2)
	spearman_coeff = 1 - 6.0 * np.sum(squared_deviation)/(N*(N*N-1))

	return spearman_coeff

if __name__=='__main__':
	filepath='ml-1m/'
	rating_matrix,validation_matrix = load_rating_matrix(filepath)
	collab_predicted_list = []
	collab_baseline_predicted_list = []
	actual_list = []

	N = validation_matrix.shape[0]

	for i in range(N):
		user_id = validation_matrix[i,0]
		movie_id = validation_matrix[i,1]
		actual_list.append(validation_matrix[i,2])

		collab_prediction = predict_rating(user_id, movie_id,rating_matrix,0)
		print(collab_prediction)
		print(validation_matrix[i,2])

		collab_baseline_prediction = predict_baseline_rating(user_id,movie_id,rating_matrix)

		collab_predicted_list.append(collab_prediction)
		collab_baseline_predicted_list.append(collab_baseline_prediction)

	print(RMSE(collab_predicted_list,actual_list))
	print(RMSE(collab_baseline_predicted_list,actual_list))
	print(Spearman_correlation(collab_predicted_list,actual_list))
	print(Spearman_correlation(collab_baseline_predicted_list,actual_list))

