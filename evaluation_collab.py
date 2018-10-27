import numpy as np
from data_parser import *
from scipy import spatial
from array import *

from prediction import *

def precision_on_top_k(validation_matrix,pred_list,good_threshold):
	'''
	This function will calculate the precision on the top k elements
	for each user and then take the average among all of them.
	USAGE:
		INPUT:
			validation_matrix	: the validation matrix containing
									the user id, movie id and actual rating
			pred_list			: the prediction made by the collaborative
									recommender system (in same sequence as
									validation data)
			good_threshold		: the threshold of the rating which we will
									use to make the relevant document set.
		OUTPUT:
			precision	: the mean precision value of each user
	'''
	#Grouping the prediction by the users
	user_dict={}
	for i in range(validation_matrix.shape[0]):
		#retreiving the actual data
		user_id=validation_matrix[i,0]
		movie_id=validation_matrix[i,1]
		actual_rating=validation_matrix[i,2]
		predict_rating=pred_list[i]

		#Actual Tuple
		actual_tuple=(movie_id,actual_rating)
		#Prediction Tuple
		pred_tuple=(movie_id,predict_rating)

		#Creating or appending the actual and prediction in user dict

		try:
			user_dict[user_id][0].append(actual_tuple)
			user_dict[user_id][1].append(pred_tuple)
		except:
			user_dict[user_id]=[[actual_tuple],[pred_tuple]]


	#Calculating the precision for each user
	overall_precision=0.0
	count=0
	for key in user_dict.keys():
		#Getting the relevant list of each user
		actual_list=user_dict[key][0]
		predict_list=user_dict[key][1]

		#Calculating the precision score
		try:
			precision=_calculate_user_precision(actual_list,predict_list,good_threshold)
			#print ("precision for user:{} is:{}".format(key,precision))

			overall_precision+=precision
			count=count+1
		except:
			continue

	#Calcualting the final precision
	overall_precision=overall_precision/count

	return overall_precision

def _calculate_user_precision(actual_list,predict_list,good_threshold):
	'''
	This function will be interanally by the precision calculation function
	for getting the precision for one user.
	USAGE:
		INPUT:
			actual_list		: the list of tuple of movieid and actual rating
			predcit_list	: the list of tuple of the movieid and the prediction
			good_threshold	: this has the same meaning as in the caller function
		OUTPUT:
			precision		: the precision for one user
	'''
	#Creating the relevant list
	#print(actual_list)
	relevant_list=[tup for tup in actual_list if tup[1]>=good_threshold]
	relevant_movie_set=set([tup[0] for tup in relevant_list])
	k_value=len(relevant_list)

	#Getting the sorted top K prediciton
	srtd_predict_list=sorted(predict_list,key=lambda x: x[1])#Sort by rating
	topk_prediction=srtd_predict_list[0:k_value]
	pred_movie_set=set([tup[0] for tup in topk_prediction])

	#Calculating the common movies in prediciton and relevant
	common_movie=relevant_movie_set.intersection(pred_movie_set)

	#Calculating the precision
	precision=len(common_movie)*1.0/k_value
	return precision

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
	filename='similarity_matrix.npz'
	rating_matrix,validation_matrix = load_rating_matrix(filepath)
	movie_sim_matrix = load_sim_matrix(filepath,filename)
	print ("Calculating the baseline matrix")
	baseline_matrix=find_baseline_matrix(rating_matrix)

	collab_predicted_list = []
	collab_baseline_predicted_list = []
	actual_list = []

	N = validation_matrix.shape[0]

	for i in range(N):
		#Taking out the validation data
		user_id = validation_matrix[i,0]
		movie_id = validation_matrix[i,1]
		actual_list.append(validation_matrix[i,2])

		#Making the Prediction
		collab_prediction = predict_rating(user_id, movie_id,rating_matrix,movie_sim_matrix,baseline_matrix,0)
		collab_baseline_prediction = predict_baseline_rating(user_id,movie_id,rating_matrix,movie_sim_matrix,baseline_matrix)
		print(validation_matrix[i,2],collab_prediction,collab_baseline_prediction)

		#Saving them to the list of results
		collab_predicted_list.append(collab_prediction)
		collab_baseline_predicted_list.append(collab_baseline_prediction)
	print(precision_on_top_k(validation_matrix,collab_predicted_list,3.0))
	print(precision_on_top_k(validation_matrix,collab_baseline_predicted_list,3.0))
	print(RMSE(collab_predicted_list,actual_list))
	print(RMSE(collab_baseline_predicted_list,actual_list))
	print(Spearman_correlation(collab_predicted_list,actual_list))
	print(Spearman_correlation(collab_baseline_predicted_list,actual_list))
