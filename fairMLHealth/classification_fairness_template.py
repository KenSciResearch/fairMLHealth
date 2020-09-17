#!/usr/bin/env python
# coding: utf-8

# # (Binary) Classification Fairness Template
# Use this template to compare fairness and performance measures across a set of trained classification models.

# Below is a recommended set of libraries. All libraries are optional unless otherwise specified

# In[ ]:


from fairMLHealth.utils import model_comparison # Required
from joblib import load
import numpy as np
import os
import pandas as pd


# # Load (Generate) Data and Models
# Here you should load (or generate) your test dataset and models.

# In[ ]:


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

# In[ ]:


# Set Pointers to be passed to 
X = None # <- add your test data
y = None # <- add your target data
protected_attr = None # add your protected attribute data
models = None # add a dict or a list of trained, scikit-compatible models


# # Generate Comparison
# 

# Comparisons can be called in one of two ways: through an object-oriented method, or through a wrapper function. The section below uses the wrapper function by default.

# In[ ]:


model_comparison.compare_models(test_data = X, target_data = y, protected_attr_data = protected_attr, models = models)

