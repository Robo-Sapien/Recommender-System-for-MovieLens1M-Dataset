import numpy as np
from data_parser import *
from collaborative import *
from evaluation_collab import *

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