#################### IMPORTING LIBRARIES ################
import numpy as np
import math
from svd import SVD
np.random.seed(1)

from data_parser import load_rating_matrix

################### HELPER FUNCTIONS ###################
class CUR():
    '''
    This class will provide one place functionality for the
    CUR decoposition along with other functions for
    making recommndation by projecting the query vector to
    concept space.
    '''
    ################# Member Variables #################
    #Training and Valdiation dataset
    data_matrix=None
    validation_matrix=None
    #SVD attributes CAUTION:(they are in ascending order)
    # sigma_vector=None
    # sigma2_vector=None
    # U_matrix=None
    # V_matrix=None
    cur_filename='cur_decomposition.npz'

    ################# Member Functions #################
    def __init__(self,filepath):
        '''
        Here we will initialize the CUR class by loading the data
        and initializing other variables.
        USAGE:
            INPUT:
                filepath    : the path where the datamatrix is stored
        '''
        #Loading the data matrix into the memory
        print ("Loading the data matrix")
        self.data_matrix,self.validation_matrix=load_rating_matrix(filepath)

        #Loading the svd decomposition into memory if present
        try:
            print ("Loading the CUR Decomposition data")
            self._load_cur_decomposition(filepath)
            print ("Load Successful")
        except:
            print ("\nLoad unsuccessful, generating the CUR decomposition")
            #Creating the eigen value decomposition
            self.generate_cur()
            print ("CUR Generation completed")
            # #Now saving the current SVD to the dataset directory
            # print ("Saving the SVD Decomposition in dataset directory")
            # self._save_cur_decomposition(filepath)

    def generate_cur(self):
        '''
        This function will generate the C, U and R matrices by
        selecting Columns and rows based on the probabilities

        USAGE:
            Call this function when using for the first time,
            to create a CUD decomposition. This will automatically
            update the C,U,R member elemets of the CUR object.
        '''
        #Getting the data matrix
        data_matrix=(self.data_matrix).astype(np.float64)

        #Calculating the probabilities for the columns.
        ColumnProb = []
        denominator = np.sum(np.square(data_matrix))
        for c in range(data_matrix.shape[1]):
            ColumnProb.append(np.sum(np.square(data_matrix[:,c]))/denominator)
        chosenColumns = np.random.choice(data_matrix.shape[1], math.floor(0.9*data_matrix.shape[1]), False, ColumnProb)
        C_matrix = np.zeros(shape=(data_matrix.shape[0],chosenColumns.shape[0]))
        i = 0
        for col in chosenColumns:
            C_matrix[:,i] = data_matrix[:,col] / math.sqrt(chosenColumns.shape[0] * ColumnProb[col])
            i += 1

        RowProb = []
        for r in range(data_matrix.shape[0]):
            RowProb.append(np.sum(np.square(data_matrix[r,:]))/denominator)
        chosenRows = np.random.choice(data_matrix.shape[0], math.floor(0.9*data_matrix.shape[0]), False, RowProb)
        R_matrix = np.zeros(shape=(chosenRows.shape[0],data_matrix.shape[1]))
        i = 0
        for row in chosenRows:
            R_matrix[i,:] = data_matrix[row,:] / math.sqrt(chosenRows.shape[0] * RowProb[row])
            i += 1

        W = np.zeros(shape=(chosenRows.shape[0], chosenColumns.shape[0]))
        for i in range(chosenRows.shape[0]):
            for j in range(chosenColumns.shape[0]):
                W[i][j] = data_matrix[chosenRows[i]][chosenColumns[j]]
        svd=SVD(None, None, 'no_normalize', 'CUR', W)
        Zplus = np.diag(1/svd.sigma_vector)
        Wplus = svd.V_matrix.dot(Zplus.dot(svd.U_matrix.T))

        reconstructed_matrix = C_matrix.dot(Wplus.dot(R_matrix))
        print(reconstructed_matrix.shape)
        


if __name__=='__main__':
    filepath='ml-1m/'
    cur=CUR(filepath)
    # sigma_vector=svd.sigma_vector
    # print sigma_vector.shape
    # for i in range(sigma_vector.shape[0]):
    #     print sigma_vector[i]
    # svd._set_90percent_energy_mode()
    # sigma_vector=svd.sigma_vector
    # print sigma_vector.shape
    # for i in range(sigma_vector.shape[0]):
    #     print sigma_vector[i]
