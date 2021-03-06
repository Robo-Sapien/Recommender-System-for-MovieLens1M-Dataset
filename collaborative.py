import numpy as np
from data_parser import *
from scipy import spatial
from operator import itemgetter
from array import *
#from collaborative_baseline import *

neighbourhood_size = 15

def generate_similarity_matrix(rating_matrix,filepath,filename):
	'''
	This function will generate the similarity matrix between
	all the movie to movie pair.
	:param rating_matrix: The utility matrix (User ID vs Movie ID)
	:param filepath: relative path to the file.
	:param filename: name of the file to be the stored.

	:return: Similarity matrix generated for the 1st time
	'''
	#Initializing the relationship matrix
	num_movies=rating_matrix.shape[1]
	sim_mat=np.zeros((num_movies,num_movies),dtype=np.float64)

	for movie1 in range(num_movies):
		for movie2 in range(movie1+1,num_movies):
			print ("Finding the similarity for {}--{}".format(movie1,movie2))
			sim_mat[movie1,movie2]=find_similarity_scores(movie1, movie2, rating_matrix)
			sim_mat[movie2,movie1]=sim_mat[movie1,movie2]

	#Saving the matrix
	filename=filepath+filename
	np.savez_compressed(filename,movie_sim_matrix=sim_mat)

	return sim_mat

def find_baseline_matrix(rating_matrix):
	'''
	This function will generate matrix containing baselines
	for all (user_id, movie_id) pairs
	:param rating_matrix: The utility matrix (User ID vs Movie ID)

	:return baseline_matrix: matrix containing baseline estimates of all possible (User id, Movie id) pairs

	'''
	#Masking the array
	non_zero_mask=rating_matrix!=0

	#Finding teh global mean
	N_rating_matrix = np.count_nonzero(rating_matrix)
	global_mean = np.sum(rating_matrix)*1.0/N_rating_matrix

	#Calculating the user rating deviation
	count_x=np.sum(non_zero_mask,axis=1,keepdims=True)
	mean_x=np.sum(rating_matrix,axis=1,keepdims=True)*1.0
	for i in range(mean_x.shape[0]):
		if count_x[i,0]!=0:
			mean_x[i,0]=mean_x[i,0]/count_x[i,0]-global_mean

	#Calculating the movie rating deviation
	count_m=np.sum(non_zero_mask,axis=0)
	mean_m=np.sum(rating_matrix,axis=0,keepdims=True)*1.0
	for i in range(mean_m.shape[1]):
		if count_m[i]!=0:
			mean_m[0,i]=mean_m[0,i]/count_m[i]-global_mean

	#print (global_mean,mean_x[0,0],mean_m[0,:])

	#Creating the baseline matrix
	dummy_ones=np.ones((rating_matrix.shape),dtype=np.float64)
	print (dummy_ones.shape)
	baseline_matrix=mean_x+(dummy_ones*mean_m)+global_mean

	return baseline_matrix

def find_similarity_scores(movie1_id, movie2_id,rating_matrix):
	'''
	This function computes the centered cosine similarity scores (Pearson correlation) for 2 movies.

	:param movie1_id: ID for 1st movie
	:param movie2_id: ID for 2nd movie
	:param rating_matrix: The utility matrix (User ID vs Movie ID)

	:return score: similarity score for given 2 movies
	'''

	movie1_ratinglist=rating_matrix[:,movie1_id]
	movie2_ratinglist=rating_matrix[:,movie2_id]

	N1 = np.count_nonzero(movie1_ratinglist)
	N2 = np.count_nonzero(movie2_ratinglist)
	if(N1==0 or N2==0):
		return 0

	mean1 = np.sum(movie1_ratinglist)*1.0/N1
	mean2 = np.sum(movie2_ratinglist)*1.0/N2

	maskedMean1 = (movie1_ratinglist>0)*mean1
	maskedMean2 = (movie2_ratinglist>0)*mean2

	centered_movie1_list = movie1_ratinglist - maskedMean1
	centered_movie2_list = movie2_ratinglist - maskedMean2

	score = 1 - spatial.distance.cosine(centered_movie1_list,centered_movie2_list)

	return score


def weighted_avg(top_similarityScore_list, neighbourhoodRating_list):
	'''
	This function computes the weighted average of neighbourhood 
	ratings w.r.t their similarity scores

	:param top_similarityScore_list: List containing top N scores
	:param neighbourhoodRating_list: List containing their corresponding ratings

	:return weightedAvg: weighted average
	'''
	top_similarityScore_list = np.asarray(top_similarityScore_list)
	neighbourhoodRating_list = np.asarray(neighbourhoodRating_list)
	numerator = np.sum(np.multiply(top_similarityScore_list, neighbourhoodRating_list))
	denominator =  np.sum(top_similarityScore_list)

	if(denominator!=0):
		weightedAvg = numerator*1.0/denominator

	return weightedAvg


if __name__=='__main__':
    filepath='ml-1m/'
    rating_matrix,validation_matrix = load_rating_matrix(filepath)

    #predictedRating = predict_rating(0,1192,rating_matrix,0)
    #print(predictedRating)
    #print(validation_matrix)
