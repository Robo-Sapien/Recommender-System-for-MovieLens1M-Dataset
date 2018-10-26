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
    reconstructed_matrix=None
    #SVD attributes CAUTION:(they are in ascending order)
    sigma_vector=None
    sigma2_vector=None
    U_matrix=None
    V_matrix=None
    svd_filename=''
    #Additional attributes for reconstruction phase
    user_mean_vec=None      #Will keep the mean rating of user
    user_var_vec=None       #will kepp the variance in the user rating
                            #this is usually standard deviation

    ################# Member Functions #################
    def __init__(self,filepath,filename,mode):
        '''
        Here we will initialize the svd class by loading the data
        and initializing other variables.
        USAGE:
            INPUT:
                filepath    : the path where the datamatrix is stored
                mode        : load/normalize, normalize the data/rating
                                matrix. if load just load the data matrix
        '''
        #Loading the data matrix into the memory
        print ("Loading the data matrix")
        self.data_matrix,self.validation_matrix=load_rating_matrix(filepath)
        #Normalizing the dataset before use
        print ("Normalizing the dataset")
        self.normalize_dataset(mode)

        #Loading the svd decomposition into memory if present
        try:
            print ("Loading the SVD Decomposition data")
            self._load_svd_decomposition(filepath,filename)
            print ("Load Successful")
        except:
            print ("\nLoad unsuccessful, generating the SVD decomposition")
            #Creating the eigen value decomposition
            self.generate_svd()
            print ("SVD Generation completed")
            #Now saving the current SVD to the dataset directory
            print ("Saving the SVD Decomposition in dataset directory")
            self._save_svd_decomposition(filepath,filename)

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

    def generate_svd(self):
        '''
        This function will generate the eigen value decomposition
        of the data matrix and store it as a member variable.

        USAGE:
            Call this function when using for the first time,
            to create a SVD decomposition. This will automatically
            update the U,Sigma,V member elemets of the SVD object.
        '''
        #Getting the data matrix (keep it here for safety)
        data_matrix=(self.data_matrix).astype(np.float64)

        #Getting the U matrix
        #Making a symmetric matrix which is positive semidefinite
        print ("Creating the U decomposition")
        aat=np.dot(data_matrix,data_matrix.T)
        #aat=aat.astype(np.float64)
        print ("Getting the eigen value decomposition")
        evalU,evecU=self._get_eigen_vectors(aat)
        self.U_matrix=evecU

        #Getting the other symmetric matrix for V
        print ("Creating the V decomposition")
        ata=np.dot(data_matrix.T,data_matrix)
        #ata=ata.astype(np.float64)
        print ("Getting the eigen value decomposition")
        evalV,evecV=self._get_eigen_vectors(ata)
        self.V_matrix=evecV

        #Removing the zero eigen values since both aat and ata
        #will have same eigen value (larger oneses has extra zeros)
        size=None
        eval=None
        if(evalU.shape[0]>evalV.shape[0]):
            #Defining the eigen value matrix
            size=evalV.shape[0]
            eval=evalV
            #Removing the trivial eigen vectors from U
            #Resizing the matrix
            self.U_matrix=self.U_matrix[:,-size:]
        else:
            size=evalU.shape[0]
            eval=evalU
            #Removing the trivial (zero) eigen vectors) from V
            #Resizing the V_matrix
            self.V_matrix=self.V_matrix[:,-size:]

        #Getting the sigma matrix from the eigen value
        print ("Creating the Sigma Diagonal Matrix")
        self.sigma2_vector=eval
        sigma=np.sqrt(np.abs(eval))
        self.sigma_vector=sigma #np.diag(sigma)

        #Now filling the U_matrix compatible with the decomposition
        self._make_U_compatible_again()

    def create_reconstruction(self,mode):
        '''
        This function will reconstruct the rating matrix, using the
        decomposed component and also denormalize it to have
        same mean and variance for each user.
        (matching their rating style)
        USAGE:
            INPUT:
                mode    : where we have to denormalize the reconstructed
                            matrix usign the mean rating of the user
        '''
        print("Reconstructing the rating-matrix")
        #Reconstructing the rating matrix
        recon=np.dot(self.U_matrix,np.diag(self.sigma_vector))
        recon=np.dot(recon,self.V_matrix.T)
        #self.reconstructed_matrix=recon

        #Since this will be a normalized matrix denormalize it
        if mode=='normalize':
            print("Renormalizing the rating-matrix")
            recon=recon*self.user_var_vec+self.user_mean_vec
        self.reconstructed_matrix=recon

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

    def _make_U_compatible_again(self):
        '''
        This function will be used internally to make the U matrix
        compatible with the overall decomposition since we will
        be having the eigen vecotrs with the ambiguous sign
        so, getting the U and V vectors separately will create problem
        '''
        print ("Making U matrix compatible again")
        self.U_matrix=np.dot(self.data_matrix,self.V_matrix)
        self.U_matrix=self.U_matrix/self.sigma_vector.reshape((1,-1))

    def _save_svd_decomposition(self,filepath,filename):
        '''
        This fucntion will save the decomposition to the dataset
        directory which can be used later without need to decomp
        -osition again.
        USAGE:
            INPUT:
                filepath    :the filepath of the dataset directory
        '''
        filename = filepath + filename
        np.savez_compressed(filename,
                            U_matrix=self.U_matrix,
                            V_matrix=self.V_matrix,
                            sigma_vector=self.sigma_vector,
                            sigma2_vector=self.sigma2_vector,)
                            #reconstructed_matrix=self.reconstructed_matrix)
        # #No need to store the data matrix now
        # self.data_matrix=None

    def _load_svd_decomposition(self,filepath,filename):
        '''
        This function will load the svd decompoased matrix to the
        member varaibles of the SVD object.
        USAGE:
            INPUT:
                filepath    :the filepath of the dataset directory
        '''
        #Loading the dataset
        filename = filepath + filename
        load_dict=np.load(filename)

        #Now assigning them to the member variables
        self.sigma_vector=load_dict['sigma_vector']
        self.sigma2_vector=load_dict['sigma2_vector']
        self.U_matrix=load_dict['U_matrix']
        self.V_matrix=load_dict['V_matrix']
        # self.reconstructed_matrix=load_dict['reconstructed_matrix']

    def _set_90percent_energy_mode(self,keep_energy=0.90):
        '''
        This function will update the eigen vectors and the eigen
        value matrices to reatin the energy only upto 90%.
        In other words this will keep the concept space to enough
        size to retain 90% of varaiance.
        '''
        #Getting the energy vector, i.e the lamdas
        print ("Starting the 90 percent energy mode")
        energy=self.sigma2_vector
        total_energy=np.sum(energy)

        #Starting to find start index leaving all previous eigen values
        start_index=0
        for i in range(energy.shape[0]):
            #Calculating the remaining after leaving the current energy
            energy_left=total_energy-energy[i]

            #Increamenting the leaving pointer
            if (energy_left/total_energy>=keep_energy):
                start_index+=1
            else:
                break

        #Now removing the insignificant energy and corresponding eigen vectors
        self.sigma_vector=self.sigma_vector[start_index:,]
        self.sigma2_vector=self.sigma2_vector[start_index:,]
        self.U_matrix=self.U_matrix[:,start_index:]
        self.V_matrix=self.V_matrix[:,start_index:]

        print ("90 percent Energy mode on")

    def make_prediction(self,userid,movieid):
        '''
        This function will make the prediction given the movieid and
        the userid by projecting the movie into the concept space
        and taking the cosine similarity of user personality
        in the same-concept space
        USAGE:
            INPUT:
                userid  : the userid for which we have to make
                            recommendation.
                movieid : the movie for which we have to make the
                            prediction for the user.
            OUPUT:
                similarity : the cosine similarity of user and movie.
                            rescale it to appropriate rating scale
                            before calculating when using
        '''
        #Extracting the user-projection in concept space (eigen movie)
        user_vector=self.U_matrix[userid-1,:]
        movie_vector=self.V_matrix[movieid-1,:]
        print(self.U_matrix.shape)
        print(self.V_matrix.shape)

        #Now Checking the similarity of user and movie
        mod_user=user_vector*user_vector
        mod_movie=movie_vector*movie_vector

        similarity=(user_vector*movie_vector)/(mod_user*mod_movie)

        return similarity

    def get_rmse_reconstruction_error(self):
        '''
        This function will calculate the root-mean squared error
        of the reconstruction and the actual matrix.
        USAGE:
            OUTPUT:
                rmse_val    : the average root mean squared error.
        '''
        non_zero_mask=self.data_matrix!=0
        diff=(self.data_matrix-self.reconstructed_matrix)*non_zero_mask
        rmse_val=np.mean(np.square(diff))**(0.5)

        #Printing the real utility matrix and prediction
        # for i in range(self.data_matrix.shape[0]):
        #     for j in range(self.data_matrix.shape[1]):
        #         if (self.data_matrix[i,j]==0):
        #             continue
        #         print ('actual:{}   prediction:{}    diff:{}'.format(
        #                 self.data_matrix[i,j],
        #                 self.reconstructed_matrix[i,j],
        #                 diff[i,j]))

        return rmse_val

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
            user_id=self.validation_matrix[i,0]-1
            movie_id=self.validation_matrix[i,1]-1

            rating_diff =   self.validation_matrix[i,2]-\
                        self.reconstructed_matrix[user_id,movie_id],
            rating_diff=np.squeeze(rating_diff)

            #Printing for interactiveness
            if i%100==0:
                print ("actual:{} prediction:{} diff:{}".format(
                    self.validation_matrix[i,2],#actual rating
                    self.reconstructed_matrix[user_id,movie_id],
                    rating_diff,
                ))

            #Storing the squared error
            squared_diff+=(rating_diff**2)

        #Taking the mean value of squared error
        rmse_val=(squared_diff/N)**(0.5)

        #Finding the spearman correlation
        pearson_coefficient=1-6*squared_diff/(N*(N*N-1))

        return rmse_val,pearson_coefficient

if __name__=='__main__':
    #Creating the svd object
    filepath='ml-1m/'
    mode='normalize'
    filename = 'svd_decomposition.npz'
    svd=SVD(filepath,filename,mode)

    #Checking the size compatibility
    print(svd.data_matrix.shape)
    print(svd.U_matrix.shape)
    print(svd.V_matrix.shape)
    print(svd.sigma_vector.shape)

    #Recosntructing the svd decomposition
    svd.create_reconstruction(mode)
<<<<<<< HEAD
    print svd.get_validation_error_correlation(),'<----vaidError'
    print svd.get_rmse_reconstruction_error(),'<----trainError'
=======
    print(svd.get_validation_error(),'<----vaidError')
    print(svd.get_rmse_reconstruction_error(),'<----trainError')
>>>>>>> 33fbaf4f47be5ee51e1d00e0688ef6710ddb63de


    # #Reconstruction with 90% eneergy left
    svd._set_90percent_energy_mode(keep_energy=0.90)
    svd.create_reconstruction(mode)
    print 'vaid:',svd.get_validation_error_correlation()
    print 'recon:',svd.get_rmse_reconstruction_error()
