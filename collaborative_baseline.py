import numpy as np
from data_parser import *
from scipy import spatial
from array import *

#from collaborative import *



def find_baseline(user_id, movie_id, rating_matrix):
	N_rating_matrix = np.count_nonzero(rating_matrix)
	global_mean = np.sum(rating_matrix)*1.0/N_rating_matrix

	user_array = rating_matrix[user_id,:]
	N_user = np.count_nonzero(user_array)
	b_x = np.sum(user_array)*1.0/N_user - global_mean

	movie_array = rating_matrix[:,movie_id]
	N_movie = np.count_nonzero(movie_array)
	b_i = np.sum(movie_array)*1.0/N_movie - global_mean

	baseline = global_mean + b_x + b_i

	return baseline




if __name__=='__main__':
    filepath='ml-1m/'
    rating_matrix,validation_matrix = load_rating_matrix(filepath)
  
    predictedRating = predict_baseline_rating(0,1192,rating_matrix)
    print(predictedRating)
