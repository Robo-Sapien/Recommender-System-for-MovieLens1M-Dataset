from collaborative import *
from collaborative_baseline import *

def load_sim_matrix(filepath,filename):
	'''
	This function will load the similarity matrix.

	:param filepath: relative path to the file.
	:param filename: name of the file to be the stored.

	:return: Similarity matrix (loaded from .npz file)


	'''
	filename=filepath+filename
	loaded=np.load(filename)

	return loaded['movie_sim_matrix']



def predict_rating(user_id, movie_id,rating_matrix,movie_sim_matrix,baseline_matrix,flag):
	'''
	This function will predict the rating according 
	to collaborative filtering algorithm.

	:param user_id: User ID for which rating has to be predicted
	:param movie_id: Movie ID for which rating has to be predicted
	:param rating_matrix: The utility matrix (User ID vs Movie ID)
	:param movie_sim_matrix: Cached matrix containing similarity scores of all possible pairs of movies
	:param baseline_matrix: matrix containing baseline estimates of all possible (User id, Movie id) pairs
	:param flag: 0 indicates normal collaborative filtering mode, 1 indicates collaborative+baseline mode

	:return: Predicted rating as per collaborative filtering algorithm
		
	'''
	movie_list_for_userid = rating_matrix[user_id,:]

	non_zero_ratings_indices_tuple = np.nonzero(movie_list_for_userid)
	non_zero_ratings_indices_array =  non_zero_ratings_indices_tuple[0]
	non_zero_ratings = movie_list_for_userid[non_zero_ratings_indices_tuple]
	non_zero_ratings = [value for value in non_zero_ratings] #list of nonzero ratings for user_id

	#flag==1 means collaborative+baseline approach mode
	if(flag==1):
		mask=rating_matrix[user_id,:]!=0
		local_baseline_list=baseline_matrix[user_id,mask]
		non_zero_ratings = np.subtract(non_zero_ratings,local_baseline_list) #subtract localBaseline to avoid double counting

	similarityScore_list=[]
	temp=[]

	for movie2_id in non_zero_ratings_indices_array:
		score=movie_sim_matrix[movie2_id,movie_id]
		similarityScore_list.append(score)

	d = dict((key, value) for (key, value) in zip(similarityScore_list,non_zero_ratings)) 

	key = d.keys() #key contains similarity scores
	key = sorted(key,reverse=True)
	values = [d[k] for k in key] #values contain corresponding ratings

	#Handling the case if neighbourhood size > size of non-zero ratings list
	iter_len=neighbourhood_size
	if iter_len>len(key):
		iter_len=len(key)

	top_similarityScore_list = [key[i] for i in range(iter_len) if key[i]>0]
	neighbourhoodRating_list = [values[i] for i in range(iter_len) if key[i]>0]

	#Handling the case if there are NO nonzero similarity scores in the topN list
	if len(top_similarityScore_list)==0 and flag==0:
		predictedRating=np.sum(rating_matrix[user_id,:])*1.0/np.count_nonzero(rating_matrix[user_id,:])
		return predictedRating #return user's average rating
	elif len(top_similarityScore_list)==0 and flag==1:
		return 0 #local estimate would be zero in baseline approach

	predictedRating = weighted_avg(top_similarityScore_list, neighbourhoodRating_list)
	return predictedRating

def predict_baseline_rating(user_id, movie_id, rating_matrix,movie_sim_matrix,baseline_matrix):
	'''
	This function will predict the rating according 
	to collaborative + baseline algorithm.

	:param user_id: User ID for which rating has to be predicted
	:param movie_id: Movie ID for which rating has to be predicted
	:param rating_matrix: The utility matrix (User ID vs Movie ID)
	:param movie_sim_matrix: Cached matrix containing similarity scores of all possible pairs of movies
	:param baseline_matrix: matrix containing baseline estimates of all possible (User id, Movie id) pairs

	:return: Predicted rating as per collaborative + baseline algorithm

	'''

	global_baseline_estimate = baseline_matrix[user_id,movie_id]
	local_baseline_estimate = predict_rating(user_id,movie_id,rating_matrix,
											movie_sim_matrix,baseline_matrix,1)

	combined_rating = global_baseline_estimate + local_baseline_estimate

	return combined_rating


if __name__=='__main__':
	filename='similarity_matrix.npz'
	filepath='ml-1m/'
	rating_matrix,validation_matrix = load_rating_matrix(filepath)

	#Loading/Creating the similarity matrix
	print ("Loading the sim matrix")
	try:
		movie_sim_matrix=load_sim_matrix(filepath,filename)
	except:
		movie_sim_matrix = generate_similarity_matrix(rating_matrix,filepath,filename)

	print ("Calculating the baseline matrix")
	baseline_matrix=find_baseline_matrix(rating_matrix)
	print(baseline_matrix[0][1192])
	print(find_baseline(0,1192,rating_matrix))
	collab_predictedRating = predict_rating(0,1192,rating_matrix,movie_sim_matrix,baseline_matrix,0)
	baseline_predictedRating = predict_baseline_rating(0,1192,rating_matrix,movie_sim_matrix,baseline_matrix)
	print(collab_predictedRating)
	print(baseline_predictedRating)
