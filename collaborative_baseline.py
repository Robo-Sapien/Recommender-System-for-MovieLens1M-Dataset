import numpy as np
from data_parser import *
from scipy import spatial
from array import *

#from collaborative import *








if __name__=='__main__':
    filepath='ml-1m/'
    rating_matrix,validation_matrix = load_rating_matrix(filepath)
  
    predictedRating = predict_baseline_rating(0,1192,rating_matrix)
    print(predictedRating)
