import numpy as np
from data_parser import *
from scipy import spatial
from operator import itemgetter
from array import *
#from collaborative_baseline import *

neighbourhood_size = 2

def find_similarity_scores(movie1_ratinglist, movie2_ratinglist):
	movie1_ratinglist=np.asarray(movie1_ratinglist)
	movie2_ratinglist=np.asarray(movie2_ratinglist)
	N1 = np.count_nonzero(movie1_ratinglist)
	N2 = np.count_nonzero(movie2_ratinglist)

	mean1 = np.sum(movie1_ratinglist)*1.0/N1
	mean2 = np.sum(movie2_ratinglist)*1.0/N2

	maskedMean1 = (movie1_ratinglist>0)*mean1
	maskedMean2 = (movie2_ratinglist>0)*mean2

	centered_movie1_list = movie1_ratinglist - maskedMean1
	centered_movie2_list = movie2_ratinglist - maskedMean2

	score = 1 - spatial.distance.cosine(centered_movie1_list,centered_movie2_list)

	return score





def weighted_avg(top_similarityScore_list, neighbourhoodRating_list):
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
  
    predictedRating = predict_rating(0,1192,rating_matrix,0)
    print(predictedRating)
    #print(validation_matrix)
