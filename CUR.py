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
    reconstructed_matrix=None
    user_mean_vec = None
    user_var_vec = None
    #CUR attributes CAUTION:(they are in ascending order)
    C_matrix = None
    U_matrix = None
    R_matrix = None

    ################# Member Functions #################
    def __init__(self,filepath,mode):
        '''
        Here we will initialize the CUR class by loading the data
        and initializing other variables.
        USAGE:
            INPUT:
                filepath    : the path where the datamatrix is stored
        '''
        self.data_matrix,self.validation_matrix=load_rating_matrix(filepath)

        #Loading the svd decomposition into memory if present
        try:
            print ("Loading the CUR Decomposition data")
            self.normalize_dataset('')
            self._load_cur_decomposition(filepath,mode)
            print ("Load Successful")
        except:
            print ("\nLoad unsuccessful, generating the CUR decomposition")
            self.normalize_dataset('normalize')
            #Creating the eigen value decomposition
            self.generate_cur(mode)
            print ("CUR Generation completed")
            # #Now saving the current CUR to the dataset directory
            print ("Saving the CUR Decomposition in dataset directory")
            self._save_cur_decomposition(filepath, mode)

    def normalize_dataset(self,mode):
        '''
        (THINK/TUNABLE)
        This function will normalize the dataset to make the non rated
        movies unbiased.(using the mean subtraction from rated movie)
        Also, to counter the high and low raters problem we have to normalize
        the rating vector of each user to bring their rating to same scale
        '''
        #Casting the dataset to floating point precision from uint8
        #self.data_matrix=np.array([[3,2,2],[2,3,-2]])
        self.data_matrix=(self.data_matrix).astype(np.float64)

        #Now calculationg the mean and varaince of each user rating
        #Getting the correct mean and variance for normalization
        non_zero_mask=(self.data_matrix!=0)

        #Calculating the mean
        num_non_zero=np.sum(non_zero_mask,axis=1,keepdims=True)
        row_sum=np.sum(self.data_matrix,axis=1,keepdims=True)

        #Creating the mean mask for getting the std of non zero rating per row
        masked_mean=non_zero_mask*row_sum
        row_diff=np.sum(np.square(self.data_matrix-masked_mean),axis=1,keepdims=True)

        self.user_mean_vec=row_sum/num_non_zero
        self.user_var_vec=(row_diff/num_non_zero)**(0.5)

        #Now normalizing the data/rating-matrix
        if mode=='normalize':
            print('normalizing the data matrix')
            #Subtracting the mean
            mask=(self.data_matrix!=0)
            masked_mean=mask*self.user_mean_vec #element wise broadcasted along user
            self.data_matrix=self.data_matrix-masked_mean
            #Diving with the varaince to brong every rating to same scale
            self.data_matrix=self.data_matrix/self.user_var_vec #element-broad user

    def generate_cur(self,mode):
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
        # data_matrix=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=np.float64)

        #Calculating the probabilities for the columns.
        ColumnProb = []
        denominator = np.sum(np.square(data_matrix))
        for c in range(data_matrix.shape[1]):
            ColumnProb.append(np.sum(np.square(data_matrix[:,c]))/denominator)
        chosenColumns = np.random.choice(data_matrix.shape[1], math.floor(0.9*data_matrix.shape[1]), False, ColumnProb)

        C_matrix = np.zeros(shape=(data_matrix.shape[0],chosenColumns.shape[0]))
        for i,col in enumerate(chosenColumns):
            C_matrix[:,i] = data_matrix[:,col] / math.sqrt(chosenColumns.shape[0] * ColumnProb[col])

        RowProb = []
        for r in range(data_matrix.shape[0]):
            RowProb.append(np.sum(np.square(data_matrix[r,:]))/denominator)
        chosenRows = np.random.choice(data_matrix.shape[0], math.floor(0.9*data_matrix.shape[0]), False, RowProb)

        R_matrix = np.zeros(shape=(chosenRows.shape[0],data_matrix.shape[1]))
        for i,row in enumerate(chosenRows):
            R_matrix[i,:] = data_matrix[row,:] / math.sqrt(chosenRows.shape[0] * RowProb[row])

        W = np.zeros(shape=(chosenRows.shape[0], chosenColumns.shape[0]))
        for i in range(chosenRows.shape[0]):
            for j in range(chosenColumns.shape[0]):
                W[i][j] = data_matrix[chosenRows[i]][chosenColumns[j]]

        svd=SVD(None, None, 'no_normalize', 'CUR', W)
        if mode == '90-percent':
            svd._set_90percent_energy_mode()
        sigma_inverse=[]
        for i in range(svd.sigma_vector.shape[0]):
            if(abs(svd.sigma_vector[i]) < 0.1):
                sigma_inverse.append(svd.sigma_vector[i])
            else:
                sigma_inverse.append(1/svd.sigma_vector[i])
        Zplus = np.diag(sigma_inverse)**2
        Wplus = svd.V_matrix.dot(Zplus.dot(svd.U_matrix.T))

        self.C_matrix = C_matrix
        self.U_matrix = Wplus
        self.R_matrix = R_matrix

        self.reconstructed_matrix = C_matrix.dot(Wplus.dot(R_matrix))
        print("Renormalizing the rating-matrix")
        self.reconstructed_matrix = self.reconstructed_matrix*self.user_var_vec+self.user_mean_vec

        non_zero_mask=self.data_matrix!=0
        diff=(self.data_matrix-self.reconstructed_matrix)*non_zero_mask
        rmse_val=np.mean(np.square(diff))**(0.5)
        print(rmse_val)

    def _save_cur_decomposition(self,filepath, mode):
        '''
        This fucntion will save the decomposition to the dataset
        directory which can be used later without need to decomp
        -osition again.
        USAGE:
            INPUT:
                filepath    :the filepath of the dataset directory
        '''
        filename = filepath + mode + 'CUR_decomposition.npz'
        np.savez_compressed(filename,
                            C_matrix=self.C_matrix,
                            U_matrix=self.U_matrix,
                            R_matrix=self.R_matrix,)
                            #reconstructed_matrix=self.reconstructed_matrix)
        # #No need to store the data matrix now
        # self.data_matrix=None

    def _load_cur_decomposition(self, filepath, mode):
        '''
        This function will load the svd decompoased matrix to the
        member varaibles of the SVD object.
        USAGE:
            INPUT:
                filepath    :the filepath of the dataset directory
        '''
        #Loading the dataset
        filename = filepath + mode + 'CUR_decomposition.npz'
        load_dict=np.load(filename)

        #Now assigning them to the member variables
        self.C_matrix=load_dict['C_matrix']
        self.U_matrix=load_dict['U_matrix']
        self.R_matrix=load_dict['R_matrix']
        # self.reconstructed_matrix=load_dict['reconstructed_matrix']
        self.reconstructed_matrix = self.C_matrix.dot(self.U_matrix.dot(self.R_matrix))
        print("Renormalizing the rating-matrix")
        self.reconstructed_matrix = self.reconstructed_matrix*self.user_var_vec+self.user_mean_vec

        non_zero_mask=self.data_matrix!=0
        diff=(self.data_matrix-self.reconstructed_matrix)*non_zero_mask
        rmse_val=np.mean(np.square(diff))**(0.5)
        print(rmse_val)

    def get_validation_error_correlation(self):
        '''
        This function will calculate the validation set error,
        by comparing the prediction made by the reconstructed matrix
        and the actal value
        USAGE:
            OUTPUT:
                rmse_val    : the root mean squared error value.
        '''
        squared_diff=0

        #Iterating over the validation examples
        N=self.validation_matrix.shape[0]
        for i in range(N):
            #Taking out the user and movie id
            user_id=self.validation_matrix[i,0]
            movie_id=self.validation_matrix[i,1]

            rating_diff =   self.validation_matrix[i,2]-\
                        self.reconstructed_matrix[user_id,movie_id],
            rating_diff=np.squeeze(rating_diff)

            #Printing for interactiveness
            # if i%100==0:
            #     print ("actual:{} prediction:{} diff:{}".format(
            #         self.validation_matrix[i,2],#actual rating
            #         self.reconstructed_matrix[user_id,movie_id],
            #         rating_diff,
            #     ))

            #Storing the squared error
            squared_diff+=(rating_diff**2)

        #Taking the mean value of squared error
        rmse_val=(squared_diff/N)**(0.5)

        #Finding the spearman correlation
        spearman_coefficient=1-6*squared_diff/(N*(N*N-1))

        return rmse_val,spearman_coefficient


if __name__=='__main__':
    filepath='ml-1m/'
    cur=CUR(filepath,'')
    print(cur.get_validation_error_correlation())

    print('\n 90% Energy: \n')
    cur=CUR(filepath,'90-percent')
    print(cur.get_validation_error_correlation())
