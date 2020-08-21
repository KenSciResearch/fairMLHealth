# fairml
Tools and libraries for fairness and bias evaluation of machine learning models for healthcare.

## Tutorial
The tutorial introduces methods and libraries for measuring fairness in machine learning models as as it relates to problems in healthcare. Through the tutorial you will first learn basic background, before generating a simple baseline model predicting Length of Stay (LOS) using data from the MIMIC-III database. This baseline model will be used as an example to understand common measures such as the Disparate Impact Ratio and Consistency Scores. You gain familiarity with the Scikit-Learn-compatible tools available in AIF360 and FairLearn, two of the most comprehensive and flexible Python libraries for measuring and addressing bias in machine learning models.

The tutorial assumes basic knowledge of machine learning implementation in Python. Before starting, please install AIF360 and FairLearn. Also, ensure that you have installed the Scipy, Pandas, Numpy, Scikit, and XGBOOST libraries. 

The tutorial also uses data from the [MIMIC III Critical Care database](https://mimic.physionet.org/gettingstarted/access/). Note that, although the data are freely available, it may take a few days to gain approval. Please save the data with the default directory name ("MIMIC"). 

