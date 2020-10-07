# fairMLHealth
Tools and tutorials for evaluation of fairness and bias in healthcare applications of machine learning models.

## Organization
This library is constructed in three main parts:
- ### Templates
    - Quickstart notebooks that serve as skeletons for your model analysis

- ### Tools
    - Methods for generating fairness comparison tables
    - Features used by templates and tutorials to facilitate comparison of multiple metrics

- ### Tutorias and Examples
    - Tutorials for measuring and analyzing fairness as it applies to machine learning
    - Examples for using the templates and tools


## Installation
Installing directly from GitHub:

    python -m pip install git+https://https://github.com/KenSciResearch/fairMLHealth

Installing from a local copy of the repo:

    pip install <path_to_fairMLHealth_dir>


## Publications
*See Publications folder for more information*

### KDD 2020 Tutorial
The KDD 2020 tutorial introduces concepts for measuring fairness in machine learning models as as it relates to problems in healthcare (slides: `publications/KDD2020-FairnessInHealthcareML-Slides.pptx`). Through the associated notebook (`publications/FairnessInhealthcareML-KDD-2020-TutorialNotebook.ipynb`) you will review the background introduced in the slides before generating a simple baseline model. This baseline will be used as an example to understand common measures such as Disparate Impact Ratio and Consistency Scores. It will also introduce you to the Scikit-Learn-compatible tools available in AIF360 and FairLearn, two of the most comprehensive and flexible Python libraries for measuring and addressing bias in machine learning models.

The tutorial notebook uses data from the [MIMIC III Critical Care database](https://mimic.physionet.org/gettingstarted/access/). Note that although the data are freely available, it may take a few days to gain approval. Please save the data with the default directory name ("MIMIC"). The notebook also requires the following Python libraries: AIF360, FairLearn, Scipy, Pandas, Numpy, Scikit, and XGBOOST. Basic knowledge of machine learning implementation in Python is assumed.


### Citation
Ahmad, M. A., Patel, A., Eckert, C., Kumar, V., & Teredesai, A. (2020, August). Fairness in Machine Learning for Healthcare. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ (pp. 3529-3530).

### Key Contributors
* Muhammad A. Ahmad
* Christine Allen
* Juhua Hu
* Carly Eckert
* Arpit Patel
* Vikas Kumar
* Ankur Teredesai
