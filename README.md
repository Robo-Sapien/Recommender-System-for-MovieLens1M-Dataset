# Recommender System
GROUP MEMBERS:

1. Shivam Bhagat        2015B5A70460H
2. Yashdeep Thorat      2015B5A70675H
3. Abhinav Kumar        2015B5A70674H



USAGE INSTRUCTION:

#################### PART 0: DATA PARSING #######################
1. First of all we have to generate the dataset in the required
    numpy format from the raw files.

2. Run python data_parser.py

3. This will create a numpy array of the dataset and store them
in the dataset directory for firthur processing.

################### PART 1: Collaborative Filtering #############
1. To first generate the similarity matrix and cache it, run:
	python prediction.py

2. Then, to make the predictions and compute errors, run:
	python evaluation_collab.py

################### PART 2: SVD DECOMPOSITION ###################
1. Simply run the following script
    python svd.py

2.This will automatically load the dataset and will load the
decomposition is already saved locally. Otherwise it will first
create the decomposition and then automatically save them
locally on the dataset directory.

################### PART 3: CUR DECOMPOSITION ###################
1. Simply run the following script
    python CUR.py

2.This will automatically load the dataset and will load the
decomposition is already saved locally. Otherwise it will first
create the decomposition and then automatically save them
locally on the dataset directory.
