from collaborative import *
from collaborative_baseline import *

def load_sim_matrix(filepath,filename):
	'''
	This function will load the similarity matrix.
	'''
	filename=filepath+filename
	loaded=np.load(filename)

	return loaded['movie_sim_matrix']



def predict_rating(user_id, movie_id,rating_matrix,movie_sim_matrix,baseline_matrix,flag):
	movie_list_for_userid = rating_matrix[user_id,:]

	non_zero_ratings_indices_tuple = np.nonzero(movie_list_for_userid)
	non_zero_ratings_indices_array =  non_zero_ratings_indices_tuple[0]
	non_zero_ratings = movie_list_for_userid[non_zero_ratings_indices_tuple]
	non_zero_ratings = [value for value in non_zero_ratings]


	if(flag==1):
		# local_baseline_list = []
		# for index in non_zero_ratings_indices_array:
		# 	local_baseline = 0#find_baseline(user_id,index,rating_matrix)
		# 	local_baseline_list.append(local_baseline)
		# local_baseline_list = np.asarray(local_baseline_list)
		# non_zero_ratings = np.subtract(non_zero_ratings,local_baseline_list)
		mask=rating_matrix[user_id,:]!=0
		local_baseline_list=baseline_matrix[user_id,mask]
		non_zero_ratings = np.subtract(non_zero_ratings,local_baseline_list)



	similarityScore_list=[]
	temp=[]

	for movie2_id in non_zero_ratings_indices_array:
		score=movie_sim_matrix[movie2_id,movie_id]
		# movie2_ratinglist = rating_matrix[:,movie2_id]
		# movie1_ratinglist = rating_matrix[:,movie_id]
		# score = find_similarity_scores(movie1_ratinglist, movie2_ratinglist)
		similarityScore_list.append(score)

	d = dict((key, value) for (key, value) in zip(similarityScore_list,non_zero_ratings))
	#print(d)
	#print(zip(similarityScore_list,non_zero_ratings))
	key = d.keys()
	key = sorted(key,reverse=True)
	values = [d[k] for k in key]

	iter_len=neighbourhood_size+1
	if iter_len>len(key):
		iter_len=len(key)

	top_similarityScore_list = [key[i] for i in range(iter_len) if key[i]>0]
	neighbourhoodRating_list = [values[i] for i in range(iter_len) if key[i]>0]

	if len(top_similarityScore_list)==0 and flag==0:
		predictedRating=np.sum(rating_matrix[user_id,:])*1.0/np.count_nonzero(rating_matrix[user_id,:])
		return predictedRating
	elif len(top_similarityScore_list)==0 and flag==1:
		return 0

	#print(temp)
	predictedRating = weighted_avg(top_similarityScore_list, neighbourhoodRating_list)
	return predictedRating

def predict_baseline_rating(user_id, movie_id, rating_matrix,movie_sim_matrix,baseline_matrix):
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

	collab_predictedRating = predict_rating(0,1192,rating_matrix,movie_sim_matrix,baseline_matrix,0)
	baseline_predictedRating = predict_baseline_rating(0,1192,rating_matrix,movie_sim_matrix,baseline_matrix)
	print(collab_predictedRating)
	print(baseline_predictedRating)
