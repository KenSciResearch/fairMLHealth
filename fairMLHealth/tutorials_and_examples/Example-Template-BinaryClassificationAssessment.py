#!/usr/bin/env python
# coding: utf-8

# # Binary Classification Fairness Assesment Template - Example
# 
# ## About
# This example is intended as a simple illustration for the use of the [Binary Classification Fairness Assessment Template](../templates/Template-BinaryClassificationAssessment.ipynb). It compares a Random Forest Classifier against fairness-aware alternative versions of that same classifier. For more information about the specific measures used, please see the [Measuring Fairness in Binary Classification Tutorial](../tutorials_and_examples/Tutorial-MeasuringFairnessInBinaryClassification.ipynb).
# 
# In the interest of simplicity, only two fairness-aware algorithms are compared in this notebook. However, several other fairness-aware models were tested in during development. For a peek at that process, see [Supplemental - Models for Binary Classification Example](../tutorials_and_examples/Supplemental-ModelsForBinaryClassificationExample.ipynb).
# 
# ## Example Contents
# 
# [Part 1](#part1) - Data Loading and Model Setup
# 
# [Part 2](#part2) - Fairness-Aware Models
# 
# [Part 3](#part3) - Model Comparison
# 

# In[37]:


from IPython.display import Markdown, HTML
from fairMLHealth.tools import reports, tutorial_helpers as helpers, model_comparison as fhmc
import numpy as np
import pandas as pd


# ----
# # Load Data and Generate Baseline Model <a name="part1"></a>

# ## MIMIC-III
# 
# This example uses a data subset from the [MIMIC-III clinical database](https://mimic.physionet.org/gettingstarted/access/) to predict "length of stay" (LOS) value. For this example, LOS is total ICU time for a given hospital admission in patients 65 and above. The raw LOS value is then converted to a binary value specifying whether an admission's length of stay is greater than the sample mean. 
# 
# Note that the code below will automatically unzip and format all necessary data for these experiments from a raw download of MIMIC-III data (saving the formatted data in the same MIMIC folder). MIMIC-III is a freely available database, however all users must pass a quick human subjects certification course. If you would like to run this example on your own, [follow these steps to be granted access to MIMIC III](https://mimic.physionet.org/gettingstarted/access/) and download the data.
# 
# 

# ## Data Subset
# 
# Data are imported at the encounter level with all additional patient identification dropped. Boolean diagnosis and procedure features are categorized through the Clinical Classifications Software system ([HCUP](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)). All features other than age are one-hot encoded and prefixed with their variable type (e.g. "GENDER_", "ETHNICITY_").  
# 

# In[2]:


# path_to_mimic_data_folder = "[path to folder containing your MIMIC-III zip files]"
path_to_mimic_data_folder = "~/data/MIMIC"


# In[42]:


# Load data and subset to ages 65+
df = helpers.load_mimic3_example(path_to_mimic_data_folder) 
df = df.loc[df['AGE'].ge(65), :]
df.drop('GENDER_F', axis = 1, inplace = True) # Redundant with GENDER_M

# Show variable count
helpers.print_feature_table(df)
display(Markdown('---'))

# Generate a binary target flagging whether an observation's length_of_stay value is above or below the mean. 
mean_val = df['length_of_stay'].mean()
df['long_los'] = df['length_of_stay'].apply(lambda x: 1 if x > mean_val else 0)
los_tbl = df[['length_of_stay', 'long_los']].describe().transpose().round(4)
tbl_style = los_tbl.style.applymap(helpers.highlight_col, 
                                    subset = pd.IndexSlice[:, 'mean'],
                                    color = "magenta"
                                  )
display(tbl_style)


# ## Split Data

# In[11]:


from sklearn.model_selection import train_test_split

# Subset and Split Data
X = df.loc[:, [c for c in df.columns 
                if c not in ['ADMIT_ID','length_of_stay', 'long_los']]]
y = df.loc[:, ['long_los']]
splits = train_test_split(X, y, stratify = y, test_size = 0.33, random_state = 42)
X_train, X_test, y_train, y_test = splits


# ## Generate Baseline
# 
# A Scikit-Learn Random Forest Classifier serves as our basis for comparison. Parameters were tuned using Scikit-Learn's GridSearch in the [Supplemental - Models for Binary Classification Example](../tutorials_and_examples/Supplemental-ModelsForBinaryClassificationExample.ipynb).

# In[12]:


from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# In[13]:


# Set model parameters (currently set as default values, but defined here to be explicit)
rf_params = {'n_estimators': 1800, 'min_samples_split': 5, 'bootstrap': False}

# Train Model
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, y_train.iloc[:, 0])
y_pred_rf = rf_model.predict(X_test)

# display performance 
print("\n", "Random Forest Prediction Scores:", "\n", 
      classification_report(y_test, y_pred_rf, 
            target_names = ['LOS <= mean', 'LOS > mean']) )


# ----
# # Fairness-Aware Models <a name="part2"></a>
# 

# ## FairLearn Models
# 
# The [FairLearn](https://fairlearn.github.io/) package includes three [mitigation algorithms](https://fairlearn.github.io/user_guide/mitigation.html) designed to increase the fairness of an existing model relative to one of two user-specified fairness metrics. Both algorithms and metrics are listed in the cell below.
# 
# For more information about the specifics of these fairness metrics, see also [Part 5 of the Measuring Fairness in Binary Classification Tutorial](../tutorials_and_examples/Tutorial-MeasuringFairnessInBinaryClassification.ipynb#part5).

# In[14]:


# Mitigation Algorithms
from fairlearn.reductions import GridSearch, ExponentiatedGradient

# Fairness Measures
from fairlearn.reductions import EqualizedOdds, DemographicParity 


# ### Fair ExponentiatedGradient
# 
# FairLearn's ExponentiatedGradient is a wrapper that runs a constrained optimization using the Exponentiated Gradient approach on a binary classification model. It treats the prediction as a sequence of cost-sensitive classification problems, returning the solution with the smallest error (constrained by the metric of choice). This approach has been demonstrated to have minimal effect on model performance by some measures. [Agarwal2018](#Agarwal2018)
# 
# This approach is applicable to sensitive attributes that are either categorical or binary/boolean. It can be used for classification problems only.
# 
# Note: solutions are not guaranteed for this approach.
# 

# In[ ]:


# Set seed for consistent results with FairLearn's ExponentiatedGradient
np.random.seed(36)  


# #### Fair ExponentiatedGradient Using Demographic Parity as Constraint

# In[17]:


eg_rfDP_model = ExponentiatedGradient(RandomForestClassifier(**rf_params), 
                                      constraints = DemographicParity()) 
eg_rfDP_model.fit(X_train, y_train,
                  sensitive_features = X_train['LANGUAGE_ENGL'])
y_pred_eg_rfDP = eg_rfDP_model.predict(X_test)

# display performance 
print("\n", "Prediction Scores:", "\n", 
      classification_report(y_test, y_pred_eg_rfDP, 
            target_names = ['LOS <= mean', 'LOS > mean']) 
      )


# #### Fair ExponentiatedGradient Using Equalized Odds as Constraint

# In[18]:


eg_rfEO_model = ExponentiatedGradient(RandomForestClassifier(**rf_params), 
                                          constraints = EqualizedOdds())  
eg_rfEO_model.fit(X_train, y_train, 
                  sensitive_features = X_train['LANGUAGE_ENGL'])
y_pred_eg_rfEO = eg_rfEO_model.predict(X_test)

# display performance 
print("\n", "Prediction Scores:", "\n", 
      classification_report(y_test, y_pred_eg_rfEO, 
            target_names = ['LOS <= mean', 'LOS > mean']) 
      )


# ### Fair GridSearch
# 
# FairLearn's GridSearch is a wrapper that runs a constrained optimization using the Grid Search approach  on a binary classification or a regression model. It treats the prediction as a sequence of cost-sensitive classification problems, returning the solution with the smallest error (constrained by the metric of choice). This approach has been demonstrated to have minimal effect on model performance by some measures. [[Agarwal2018]](#Agarwal2018)
# 
# This approach is applicable to sensitive attributes that are binary/boolean only. It can be used for either binary classification or regression problems.
# 

# #### Fair GridSearch Using Equalized Odds as Constraint

# In[19]:


# Train GridSearch
gs_rfEO_model = GridSearch(RandomForestClassifier(**rf_params),
                           constraints = EqualizedOdds(),
                           grid_size = 45)

gs_rfEO_model.fit(X_train, y_train, 
                  sensitive_features = X_train['LANGUAGE_ENGL'])
y_pred_gs_rfEO = gs_rfEO_model.predict(X_test)

# display performance 
print("\n", "Prediction Scores:", "\n", 
      classification_report(y_test, y_pred_gs_rfEO, 
            target_names = ['LOS <= mean', 'LOS > mean']) 
      )


# #### Fair GridSearch Using Demographic Parity as Constraint

# In[20]:


# Train GridSearch
gs_rfDP_model = GridSearch(RandomForestClassifier(**rf_params),
                           constraints = DemographicParity(),
                           grid_size = 45)

gs_rfDP_model.fit(X_train, y_train, 
                  sensitive_features = X_train['LANGUAGE_ENGL'])
y_pred_gs_rfDP = gs_rfDP_model.predict(X_test)

# display performance 
print("\n", "Prediction Scores:", "\n", 
      classification_report(y_test, y_pred_gs_rfDP, 
            target_names = ['LOS <= mean', 'LOS > mean']) 
      )


# ----
# # Model Comparison <a name="part3"></a>
# 

# ## Set the Required Variables  
# 
# * X (numpy array or similar pandas object): test data to be passed to the models to generate predictions. It's recommended that these be separate data from those used to train the model.
# 
# * y (numpy array or similar pandas object): target data array corresponding to X. It is recommended that the target is not present in the test_data.
# 
# * models (list or dict-like): the set of trained models to be evaluated. Note that the dictionary keys are assumed as model names. If a list-like object is passed, the function will set model names relative to their index (i.e. "model_0", "model_1", etc.)
# 
# * protected_attr (numpy array or similar pandas object): protected attributes correspoinding to X, optionally also included in X. Note that values must currently be binary- or boolean-type.
# 

# In[21]:


X = X_test
y = y_test
protected_attr = X_test['LANGUAGE_ENGL']
models = {'rf_model': rf_model,
         'gs_rfEO_model': gs_rfEO_model, 'gs_rfDP_model': gs_rfDP_model,
         'eg_rfEO_model': eg_rfEO_model, 'eg_rfDP_model': eg_rfDP_model}
print("Models being compared in this example:", list(models.keys()))


# ## Comparison with the FairMLHealth Tool
# 
# The FairMLHealth model comparison tool generates a table of fairness measures that can be used to quickly compare the fairness-performance tradeoff for a set of fairness-aware models. 
# 
# Note that there is some additional formatting added to the cell below simply to add highlighting for this example

# In[32]:


from importlib import reload
reload(reports)


# In[36]:


# Generate comparison table (returned as a pandas dataframe)
comparison = fhmc.compare_models(X, y, protected_attr, models)

# Here we determine the indices for equal odds measures so that we can highlight according
#    to those indices later
idx = pd.IndexSlice
eotag = idx[:, ['Equal Opportunity Difference', 'Equalized Odds Difference',
                 'Equalized Odds Ratio']
            ]
equal_odds = comparison.loc[eotag, :].index

# Here we return the flagged table as a pandas styler so we can also highlight 
#       measures of Equal Odds
# Note that the HTML wrapper around the *.render() method facilitates color rendering for
#     GitHub, and is not necessary to view the colors natively in jupyter
table = reports.flag_suspicious(comparison, as_styler = True
            ).apply(lambda x: ['background-color:teal' 
                                if x.name in equal_odds else '' for i in x]
                    , axis = 1)
HTML(table.render())


# ## Comparison with the FairLearn Dashboard
# 
# FairLearn comes with its own model comparison dashboard to allow visual comparison between models.
# 

# In[38]:


from fairlearn.widget import FairlearnDashboard

# FairLearnDashboard Note: for binary classification models, arrays must be passed as a list
FairlearnDashboard(sensitive_features = protected_attr.to_list(), 
                   sensitive_feature_names = ['LANGUAGE_ENGL'],
                   y_true = y.iloc[:, 0].to_list(),
                   y_pred = {k:model.predict(X) for k,model in models.items()})


# # References
# 
# <a name="Agarwal2018"></a>
# Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. [rXiv preprint arXiv:1803.02453](https://arxiv.org/pdf/1803.02453.pdf).
