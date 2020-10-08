#!/usr/bin/env python
# coding: utf-8

# # (Binary) Classification Fairness Template - Example
# Use this template to compare fairness and performance measures across a set of trained binary classification models.

# In[1]:


from fairMLHealth.utils import model_comparison # Required
from joblib import load
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message="Pass n_neighbors=5 as keyword args. From version 0.25")


# # Load (Generate) Data and Models
# Here you should load (or generate) your test dataset and models.

# In[2]:


# < Optional Loading/Cleanin?g/Training Code Here >


# ## Set the Required Variables  
# 
# * X (numpy array or similar pandas object): test data to be passed to the models to generate predictions. It's recommended that these be separate data from those used to train the model.
# 
# * y (numpy array or similar pandas object): target data array corresponding to X. It is recommended that the target is not present in the test_data.
# 
# * models (list or dict-like): the set of trained models to be evaluated. Note that the dictionary keys are assumed as model names. If a list-like object is passed, the function will set model names relative to their index (i.e. "model_0", "model_1", etc.)
# 
# * protected_attr (numpy array or similar pandas object): protected attributes correspoinding to X, optionally also included in X. Note that values must currently be binary- or boolean-type.

# In[3]:


# Validate that example data are present
data_file = os.path.expanduser("~/data/fairness_and_bias/mimic_model_comparison/binary_classification.joblib")
if not os.path.isfile(data_file):
    raise ValueError(f"MIMIC data could not be loaded from {path_to_mimic_data_folder}. "+
                     "Please run the \"Example Models for Classification Template\" Notebook")


# In[4]:


# Example loading data from an external notebook. Data and models were saved as one object.
input_data = load(data_file)
X = input_data.X
y = input_data.y
models = input_data.models
print("Models available in this example:", list(models.keys()))


# ## Example of Protected Attributes
# Protected attributes are currently required to be binary- or boolean-type values. You may include one or several attributes in the analysis, however it is recommended that you start with one. By convention, members of the privileged group should be labeled as 1 (or True), with unprivileged members labeled as a 0 (False).
# 
# Below is an example of how you might generate such a variable.

# In[5]:


# Generate indicator for protected attribute
lang_cols = [c for c in X.columns if c.startswith("LANGUAGE_")]
eng_cols = ['LANGUAGE_ENGL']
X_lang =  X.loc[:,lang_cols]
english_speaking = X[eng_cols].eq(1).any(axis=1)
protected_attr = english_speaking.astype(int)
protected_attr.name = 'ENGLISH_SPEAKING'


# # Generate Comparison
# 

# Comparisons can be called in one of two ways: through an object-oriented method, or through a wrapper function. The section below demonstrates the object-oriented version - first showing results for a single model, then showing results for the full group of models.

# In[6]:


comp = model_comparison.fairCompare(test_data=X, target_data=y, protected_attr_data=protected_attr, models=models)
comp.measure_model('naive_bayes_model')


# In[7]:


comp.compare_models()


# Below is an example of the wrapper function

# In[8]:


model_comparison.compare_models(X, y, protected_attr, models={'naive_bayes_model':models['naive_bayes_model']})


# In[ ]:




