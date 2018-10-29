.. RecommenderSystem documentation master file, created by
   sphinx-quickstart on Fri Oct 26 23:12:24 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Recommender System Documentation
*********************************

Usage Information
=================

PART 0: DATA PARSING

    1. First of all we have to generate the dataset in the required
        numpy format from the raw files.

    2. Run python data_parser.py

    3. This will create a numpy array of the dataset and store them
        in the dataset directory for firthur processing.

PART 1: Collaborative Filtering

    1. To first generate the similarity matrix and cache it, run:
    	python prediction.py

    2. Then, to make the predictions and compute errors, run:
    	python evaluation_collab.py

PART 2: SVD DECOMPOSITION

    1. Simply run the following script
        python svd.py

    2.This will automatically load the dataset and will load the
        decomposition is already saved locally. Otherwise it will first
        create the decomposition and then automatically save them
        locally on the dataset directory.

PART 3: CUR DECOMPOSITION

    1. Simply run the following script
        python CUR.py

    2.This will automatically load the dataset and will load the
        decomposition is already saved locally. Otherwise it will first
        create the decomposition and then automatically save them
        locally on the dataset directory.

Documentation for the Code
***************************
.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

Recommender System Collaborative
==================================
1. collaborative.py
    .. automodule:: collaborative
       :members:
2. evaluation_collab.py
    .. automodule:: evaluation_collab
      :members:
3. prediction.py
    .. automodule:: prediction
     :members:

Recommender System SVD
========================
.. automodule:: svd
   :members:

Recommender System CUR
========================
.. automodule:: CUR
  :members:

Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
