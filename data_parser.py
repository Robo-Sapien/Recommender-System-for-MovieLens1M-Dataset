import numpy as np
import pandas as pd

def generate_rating_matrix(filepath,filename='ratings.dat'):
    '''
    This function will genereate the user-movie rating matrix.
    USAGE:
        INPUT:
            filepath     : the filepath to the dataset
            filename     : the rating filename
        OUTPUT:
            rating_matrix: a numpy matrix of type integer(np.int8)

        This function will automatically save the rating matrix in
        dataset directory. Please us that saved matrix for use later.

    '''
    #Creating a numpy matrix
    rating_matrix=np.zeros(shape=(6040,3952),dtype=np.int8)

    #Iterating over the rating to create the rating matrix
    filename=filepath+filename
    with open(filename) as fhandle:
        for line in fhandle:
            #Extracting the fields in the rating line
            fields=line.split('::')
            userid=int(fields[0])
            movieid=int(fields[1])
            rating=int(fields[2])

            print "Filling user {}:movie {}:rating {}".format(userid,movieid,
                                                            rating)
            rating_matrix[userid-1,movieid-1]=rating

    #Saving the numpy array in compressed format
    matrix_filename=filepath+'rating_matrix.npz'
    np.savez_compressed(matrix_filename,rating_matrix=rating_matrix)

    return rating_matrix

def load_rating_matrix(filepath,filename='rating_matrix.npz'):
    '''
    This function will load the user rating matrix stared as
    compressed numpy array.
    USAGE:
        INPUT:
            filepath    : the location of the file
            filename    : the name of the numpy saved file
        OUTPUT:
            rating_matrix: the rating matrix
    '''
    #loading the numpy dictionary
    arrays=np.load(filepath+filename)

    #Return the array
    return arrays['rating_matrix']

if __name__=='__main__':
    filepath='ml-1m/'
    filename='ratings.dat'
    generate_rating_matrix(filepath,filename)
