#################### IMPORTING LIBRARIES ################
import numpy as np
np.random.seed(1)

from data_parser import load_rating_matrix

################### HELPER FUNCTIONS ###################
class SVD():
    '''
    This class will provide one place functionality for the
    singular valued decoposition along with other functions for
    making recommndation by projecting the query vector to
    concept space.
    '''
    ################# Member Variables #################
    #Training and Valdiation dataset
    data_matrix=None
    validation_matrix=None
    #SVD attributes
    sigma_vector=None
    sigma2_vector=None
    U_matrix=None
    V_matrix=None
    svd_filename='svd_decomposition.npz'

    ################# Member Functions #################
    def __init__(self,filepath):
        '''
        Here we will initialize the svd class by loading the data
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
            print ("Loading the SVD Decomposition data")
            self._load_svd_decomposition(filepath)
            print ("Load Successful")
        except:
            print ("\nLoad unsuccessful, generating the SVD decomposition")
            #Creating the eigen value decomposition
            self.generate_svd()
            print ("SVD Generation completed")
            #Now saving the current SVD to the dataset directory
            print ("Saving the SVD Decomposition in dataset directory")
            self._save_svd_decomposition(filepath)

    def generate_svd(self):
        '''
        This function will generate the eigen value decomposition
        of the data matrix and store it as a member variable.

        USAGE:
            Call this function when using for the first time,
            to create a SVD decomposition. This will automatically
            update the U,Sigma,V member elemets of the SVD object.
        '''
        #Getting the data matrix
        data_matrix=(self.data_matrix).astype(np.float64)

        #Getting the U matrix
        #Making a symmetric matrix which is positive semidefinite
        print ("Creating the U decomposition")
        aat=np.dot(data_matrix,data_matrix.T)
        #aat=aat.astype(np.float64)

        print ("Getting the eigen value decomposition")
        eval,evecU=self._get_eigen_vectors(aat)
        self.U_matrix=evecU

        #Getting the sigma matrix from the eigen value
        print ("Creating the Sigma Diagonal Matrix")
        self.sigma2_vector=eval
        sigma=np.sqrt(np.abs(eval))
        self.sigma_vector=sigma #np.diag(sigma)

        #Getting the other symmetric matrix for V
        print ("Creating the V decomposition")
        ata=np.dot(data_matrix.T,data_matrix)
        #ata=ata.astype(np.float64)
        print ("Getting the eigen value decomposition")
        _,evecV=self._get_eigen_vectors(ata)
        self.V_matrix=evecV

    def _get_eigen_vectors(self,K):
        '''
        This function will be used for computation of the eigen value
        and eigen vectors of a symmetric matrix.
        USAGE:
            INPUT:
                K       : a symmetric matrix
            OUTPUT:
                eval    : the eigen value vector in ascending order
                evec    : the eigen vector matrix as a columns in same
                            order as the eigen value.
        '''
        eval,evec=np.linalg.eigh(K)

        return eval,evec

    def _save_svd_decomposition(self,filepath):
        '''
        This fucntion will save the decomposition to the dataset
        directory which can be used later without need to decomp
        -osition again.
        USAGE:
            INPUT:
                filepath    :the filepath of the dataset directory
        '''
        filename=filepath+self.svd_filename
        np.savez_compressed(filename,
                            U_matrix=self.U_matrix,
                            V_matrix=self.V_matrix,
                            sigma_vector=self.sigma_vector,
                            sigma2_vector=self.sigma2_vector)
        # #No need to store the data matrix now
        # self.data_matrix=None

    def _load_svd_decomposition(self,filepath):
        '''
        This function will load the svd decompoased matrix to the
        member varaibles of the SVD object.
        USAGE:
            INPUT:
                filepath    :the filepath of the dataset directory
        '''
        #Loading the dataset
        filename=filepath+self.svd_filename
        load_dict=np.load(filename)

        #Now assigning them to the member variables
        self.sigma_vector=load_dict['sigma_vector']
        self.sigma2_vector=load_dict['sigma2_vector']
        self.U_matrix=load_dict['U_matrix']
        self.V_matrix=load_dict['V_matrix']

    #def _set
if __name__=='__main__':
    filepath='ml-1m/'
    svd=SVD(filepath)
    sigma_vector=svd.sigma_vector
    for i in range(sigma_vector.shape[0]):
        print sigma_vector[i]
