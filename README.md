# fairMLHealth
Tools and libraries for fairness and bias evaluation of machine learning models for healthcare.

## Tutorial
oThe tutorial introduces concepts for measuring fairness in machine learning models as as it relates to problems in healthcare (slides: `publications/FairnessInHealthcareML-KDD-2020.pptx`). Through the associated notebook (`fairMLHealth/tutorial_and_examples/kdd_fairness_in_healthcare_tutorial.ipynb`) you will review the background introduced in the slides before generating a simple baseline model. This baseline will be used as an example to understand common measures such as Disparate Impact Ratio and Consistency Scores. It will also introduce you to the Scikit-Learn-compatible tools available in AIF360 and FairLearn, two of the most comprehensive and flexible Python libraries for measuring and addressing bias in machine learning models.

The tutorial assumes basic knowledge of machine learning implementation in Python.

The tutorial notebook uses data from the [MIMIC III Critical Care database](https://mimic.physionet.org/gettingstarted/access/). Note that although the data are freely available, it may take a few days to gain approval. Please save the data with the default directory name ("MIMIC"). The notebook also requires the following Python libraries: AIF360, FairLearn, Scipy, Pandas, Numpy, Scikit, and XGBOOST.

## Citation
Ahmad, M. A., Patel, A., Eckert, C., Kumar, V., & Teredesai, A. (2020, August). Fairness in Machine Learning for Healthcare. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ (pp. 3529-3530).

