#!/usr/bin/env python
# coding: utf-8

# # Binary Classification Fairness Assessment Template
# 
# Use this template as a skeleton for comparing fairness and performance measures across a set of trained binary classification models. For an example with a completed comparison, see [tutorials_and_examples/Example-Template-BinaryClassificationAssessment.ipynb](../tutorials_and_examples/Example-Template-BinaryClassificationAssessment.ipynb).

# In[ ]:


# Recommended list of libraries (optional unless otherwise specified)
from fairMLHealth.tools import model_comparison as fhmc # Required
import os
import pandas as pd


# ----
# # Load (or Generate) Data and Models
# 
# Here you should load (or generate) your test dataset and models.

# In[ ]:


# < Optional Loading/Cleaning/Training Code Here >


# ----
# # Compare Models
# 

# ## Required Variables  
# 
# * X (numpy array or similar pandas object): test data to be passed to the models to generate predictions. It's recommended that these be separate data from those used to train the model.
# 
# * y (numpy array or similar pandas object): target data array corresponding to X. It is recommended that the target is not present in the test_data.
# 
# * models (list or dict-like): the set of trained models to be evaluated. Note that the dictionary keys are assumed as model names. If a list-like object is passed, the function will set model names relative to their index (i.e. "model_0", "model_1", etc.)
# 
# * protected_attr (numpy array or similar pandas object): protected attributes correspoinding to X, optionally also included in X. Note that values must currently be binary- or boolean-type.

# In[ ]:


# Set Pointers to be Passed to the Comparison Tools
X = None # <- add your test data
y = None # <- add your test labels
protected_attr = None # add your protected attribute data
models = None # add a dict or a list of trained, scikit-compatible models


# ## Comparison with the FairMLHealth Tool
# 
# The FairMLHealth model comparison tool generates a table of fairness measures that can be used to quickly compare the fairness-performance tradeoff for a set of fairness-aware models. Comparisons can be called in one of two ways: through an object-oriented method, or through a wrapper function. The section below uses the wrapper function by default.

# In[ ]:


fhmc.compare_models(test_data = X, 
                    target_data = y, 
                    protected_attr_data = protected_attr, 
                    models = models)


# ## Comparison with the FairLearn Dashboard
# 
# FairLearn comes with its own model comparison dashboard to allow visual comparison between models.  Note that for binary classification models the ground truth must be passed as a list.

# In[ ]:


from fairlearn.widget import FairlearnDashboard

# FairLearnDashboard Note: for binary classification models, arrays must be passed as a list
FairlearnDashboard(sensitive_features=protected_attr.to_list(), 
                   sensitive_feature_names=['LANGUAGE_ENGL'],
                   y_true=y.iloc[:,0].to_list(),
                   y_pred={k:model.predict(y) for k,model in models.items()})

