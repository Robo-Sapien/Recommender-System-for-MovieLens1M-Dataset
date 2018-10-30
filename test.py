import numpy as np
from data_parser import *
from collaborative import *
from evaluation_collab import *
from prediction import *

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
#start1 = time.time()
for i in range(N):
	#Taking out the validation data
	user_id = validation_matrix[i,0]
	movie_id = validation_matrix[i,1]
	actual_list.append(validation_matrix[i,2])

	#Making the Prediction

	collab_prediction = predict_rating(user_id, movie_id,rating_matrix,movie_sim_matrix,baseline_matrix,0)


	collab_baseline_prediction = predict_baseline_rating(user_id,movie_id,rating_matrix,movie_sim_matrix,baseline_matrix)

	#print(validation_matrix[i,2],collab_prediction,collab_baseline_prediction)

	#Saving them to the list of results
	collab_predicted_list.append(collab_prediction)
	collab_baseline_predicted_list.append(collab_baseline_prediction)

#print "Time taken for 1 collaborative prediction", time_collab
#print "Time taken for 1 collaborative+baseline prediction", time_collab
print("RMSE for collaborative: ",RMSE(collab_predicted_list,actual_list))
print("RMSE for collaborative+baseline: ",RMSE(collab_baseline_predicted_list,actual_list))


'''
filepath='ml-1m/'
rating_matrix,validation_matrix = load_rating_matrix(filepath)

arr=rating_matrix[0,:]
print(rating_matrix)
print(arr)
arr=np.nonzero(arr)
#np.asarray(arr)
print(type(arr[0]))
print(np.sum(arr[0]))

for value in arr[0]:
	print(value)

list=[1,2,3]
print(list[0])
print(validation_matrix)

print(find_similarity_scores(0,1192,rating_matrix))

print(RMSE([1,2],[1,4]))

print(Spearman_correlation([1,4],[1,2]))
'''