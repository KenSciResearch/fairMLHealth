#!/usr/bin/env python
# coding: utf-8

# # Tutorial of Fairness Metrics for Healthcare
# 
# This tutorial introduces methods and libraries for measuring fairness and bias in machine learning models as as they relate to problems in healthcare. After providing some background, it will generate a simple baseline model predicting Length of Stay (LOS) using data from the [MIMIC-III database](https://mimic.physionet.org/gettingstarted/access/). It will then use variations of that model to demonstrate common measures of "fairness" using [AIF360](http://aif360.mybluemix.net/), a prominent library for this purpose, before comparing AIF360 to another prominent library, [FairLearn](https://fairlearn.github.io/).
# 

# ### Tutorial Contents
# [Part 0:] Background
# 
# [Part 1:](#part1) Model Setup
# 
# [Part 2:](#part2) Metrics of Fairness in AIF360
# 
# [Part 3:](#part3) Comparing Against a Second Model - Evaluating Unawarenes
# 
# [Part 4:](#part4) Testing Other Sensitive Attributes
# 
# [Part 5:](#part5) Comparison to FairLearn
# 
# ### Requirements
# This tutorial assumes basic knowledge of machine learning implementation in Python. Before starting, please install [AIF360](http://aif360.mybluemix.net/) and [FairLearn](https://fairlearn.github.io/). Also, ensure that you have installed the Pandas, Numpy, Scikit, and XGBOOST libraries.

# ## Part 0: Background 
# SECTIONS TO BE INCLUDED:
# * what is fairness
# * metrics for fairness
# * list of measures that will be included in this notebook

# ## Part 1: Model Setup <a class="anchor" id="part1"></a>
# 
# This section introduces and loads the data subset that will be used in this tutorial. Then it generates a simple baseline model to be used throughout the tutorial.

# In[4]:


import numpy as np
import os
import pandas as pd
import sys

# Jupyter Add-Ons from local folder
import tutorial_helpers

# Prediction Libs
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# Metrics
from aif360.sklearn.metrics import *
from fairlearn.metrics import (
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    balanced_accuracy_score_group_summary, roc_auc_score_group_summary,
    equalized_odds_difference, difference_from_summary)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


# ### MIMIC III
# This tutorial uses data from the MIMIC III Critical Care database, a freely accessible source of Electronic Health Records from Beth Israel Deaconess Medical Center in Boston, years 2001 through 2012. To download the MIMIC III data, please use this link: [Access to MIMIC III](https://mimic.physionet.org/gettingstarted/access/). Please save the data in a folder with the default name ("MIMIC").
# 
# The raw MIMIC download contains only a folder of zipped_files. The tutorial code will automatically unzip and format the necessary data for this experiment, saving the formatted data in the current folder. Simply enter the correct path of the MIMIC folder in the following cell to enable this feature. Your path should end with the directory "MIMIC".
# 
# Example: path_to_mimic_data_folder = "~/data/MIMIC"

# In[5]:


# path_to_mimic_data_folder = "[path to your downloaded data folder]"
path_to_mimic_data_folder = "~/data/MIMIC"


# ### Data Subset
# This example uses data from all years of the MIMIC data for patients aged 65 and older. Features include diagnosis and procedure codes categorized through the Clinical Classifications Software system ([HCUP](#hcup)). 
# 
# Data are imported at the encounter level, with patient identification dropped. All features are one-hot encoded and prefixed with their variable type (e.g. "GENDER_", "ETHNICITY_"). 

# In[6]:


df = tutorial_helpers.load_example_data(path_to_mimic_data_folder) # note: ADMIT_ID has been masked
df.head()


# ### Baseline Length of Stay Model
# For this simple model, we'll predict the length of time spent in the ICU.
# 
# Two target variables will be used for the following experiments: 'length_of_stay' and 'los_binary'. For this dataset,  length_of_stay is, of course, the true value of the length of the patient's stay in days. The los_binary variable is a binary variable indicating whether the admission resulted in a length of stay either < or >= the mean.

# In[7]:


mean_val=df['length_of_stay'].mean()
df['los_binary'] = df['length_of_stay'].apply(lambda x: 0 if x<=mean_val else 1)
df[['length_of_stay', 'los_binary']].describe().round(4)


# In[8]:


# Subset and split data for the first model
X = df.loc[:,['ADMIT_ID']+[c for c in df.columns if (c.startswith('AGE') or c.startswith('DIAGNOSIS_') or c.startswith('PROCEDURE_'))]]
y = df.loc[:, ['los_binary']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# generate alternative model
baseline_model = XGBClassifier()
baseline_model.fit(X_train, y_train)
baseline_y_pred = baseline_model.predict(X_test)

#
print('\n', "Baseline ROC_AUC Score:", roc_auc_score(y_test, baseline_y_pred) )


# ## Part 2: Testing Gender as a Sensitive Attribute <a class="anchor" id="part2"></a>
# Our first experiment will test the effect of including the sensitive attribute 'GENDER_M'. This attribute is encoded in our data as a boolean attribute, where 0=female and 1=male, since males are assumed to be the privileged group. For the purposes of this experiment all other senstitive attributes and potential proxies will be dropped, such that only gender, diangosis, and procedure codes will be used to make the prediction.
# 
# At first we will examine the measurements for each approach on their own. 
# 
# We will see that while some measures can be used to analyze the model in isolation, others require comparison against other metrics.

# In[9]:


df.groupby('GENDER_M')['length_of_stay'].describe().round(4)


# In[10]:


# Generate a model that includes gender as a feature
X_train_gender = X_train.join(df[['GENDER_M']], how='inner')
X_test_gender = X_test.join(df[['GENDER_M']], how='inner')

model = XGBClassifier()
model.fit(X_train_gender, y_train)
y_pred_gender = model.predict(X_test_gender)

#
print('\n', "ROC_AUC Score with Gender Included:", roc_auc_score(y_test, y_pred_gender) )


# ### Measuring Fairness via AIF360
# 
# AIF360 requires the sensitive attribute to be in the same dataframe (or 2-D array) as the target variable (both the ground truth and the prediction), so we add that here
# 

# In[11]:


y_test_aif = pd.concat([X_test_gender['GENDER_M'], y_test], axis=1).set_index('GENDER_M')
y_pred_aif = pd.concat([X_test_gender['GENDER_M'].reset_index(drop=True), pd.Series(y_pred_gender)], axis=1).set_index('GENDER_M')
y_pred_aif.columns = y_test_aif.columns


# ### Prediction Rates
# The base rate is the average value of the ground truth (optionally weighted). It provides useful context, although it is not technically a measure of fairness. 
# > $base\_rate = \sum_{i=0}^N(y_i)/N$
# 
# The Selection Rate is the average value of the predicted (ŷ).
# > $selection\_rate = \sum_{i=0}^N(ŷ_i)/N$

# In[12]:


model_scores =  pd.DataFrame(columns=('measure','value'))
print("base_rate:", round(base_rate(y_test_aif, y_pred_aif), 4), "\n")
model_scores.loc[0] = ['selection_rate', selection_rate(y_test_aif, y_pred_aif)]
print(model_scores)


# ### Measures of Demographic Parity
# 
# The Disparate Impact Ratio is the ratio between the probability of positive prediction for the unprivileged group and the probability of positive prediction for the privileged group. A ratio of 1 indicates that the model is fair (it favors neither group).
# > $disparate\_impact\_ratio = P(ŷ =1 | unprivileged) / P(ŷ =1 | privileged)$
# 
# Statistical Parity Difference is the difference between the selection rate of the privileged group and that of the unprivileged group. A difference of 0 indicates that the model is fair (it favors neither group).
# > $statistical\_parity\_difference = selection\_rate_{unprivileged} - selection\_rate_{privileged} $

# In[13]:


model_scores.loc[1] = ['disparate_impact_ratio', disparate_impact_ratio(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.loc[2] = ['statistical_parity_difference', statistical_parity_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.tail(2)


# ### Measures of Equal Odds
# Average Odds Difference is the average of the difference in FPR and TPR for the unprivileged and privileged groups.
# > to do: add equation and better explanation 
# 
# Average Odds Error is the average of the absolute difference in FPR and TPR for the unprivileged and privileged groups.
# > to do: add equation and better explanation 
# 
# Equal Opportunity Difference is the difference in recall scores (TPR) between the unprivileged and privileged groups.
# > to do: add equation and better explanation 

# In[14]:


model_scores.loc[3] = ['average_odds_difference', average_odds_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.loc[4] = ['average_odds_error', average_odds_error(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.loc[5] = ['equal_opportunity_difference', equal_opportunity_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.tail(3)


# ### Measures Of Individual Fairness
# Consistency scores measure the similarity between a given prediction and the predictions of "like" individuals. In AIF360, the consistency score is calculated as the compliment of the mean distance to the score of the mean nearest neighbhor, using Scikit's Nearest Neighbors algorithm (default 5 neighbors determined by BallTree algorithm).
# > $ consistency\_score = 1 - |mean_{distance}({5\ nearest\ neighbors})| $
# 
# #### The Generalized Entropy Index and Related Measures
# The Generalized Entropy (GE) Index is...
# > to do: add equation and better explanation 
# 
# Generalized Entropy Error = Calculates the GE of the set of errors, i.e. 1 + (ŷ == pos_label) - (y == pos_label) 
# > to do: format equation and add better explanation 
# 
# Between Group Generalized Entropy Error = Calculates the GE of the set of mean errors for the two groups (privileged error & unprivileged error), weighted by the number of predictions in each group
# > to do: add equation and better explanation 

# In[15]:


model_scores.loc[6] = ['consistency_score', consistency_score(X_test_gender, y_pred_gender)]
model_scores.loc[7] = ['generalized_entropy_error', generalized_entropy_error(y_test['los_binary'], y_pred_gender)]
model_scores.loc[8] = ['between_group_generalized_entropy_error', 
                            between_group_generalized_entropy_error(y_test_aif, y_pred_aif, prot_attr=['GENDER_M'])]
model_scores.tail(3) 


# ## Part 3: Comparing Against a Second Model - Evaluating Unawareness <a class="anchor" id="part3"></a>
# 
# To demonstrate the change in model scores relative to the use of a sensitive attribute, this section generates a new, though similar model with the sensitive attribute removed. As shown below...
# > to do: add takeaway
# 
# Note: Since we have already discussed the individual measures, a helper function will be used to save space.

# In[26]:


new_scores = tutorial_helpers.get_aif360_measures_df(X_test_gender, y_test, baseline_y_pred, sensitive_attributes=['GENDER_M'])

comparison = model_scores.rename(columns={'value':'gender_score'}
                                ).merge(new_scores.rename(columns={'value':'gender_score (feature removed)'}))
comparison.round(4)


# ## Part 4: Testing Other Sensitive Attributes
# 
# Our next experiment will test the presence of bias relative to a patient\'s language, assuming that there is a bias toward individuals who speak English. As above, we will add a boolean 'LANGUAGE_ENGL' feature to the baselie data.
# 
# > to do: fix broken HTML anchor

# In[18]:


# Here we attach the sensitive attribute to our data
lang_cols = [c for c in df.columns if c.startswith("LANGUAGE_")]
eng_cols = ['LANGUAGE_ENGL']
X_lang =  df.loc[:,lang_cols]
X_lang['LANG_ENGL'] = 0
X_lang.loc[X_lang[eng_cols].eq(1).any(axis=1), 'LANG_ENGL'] = 1
X_lang = X_lang.drop(lang_cols, axis=1).fillna(0)
X_lang.join(df['length_of_stay']).groupby('LANG_ENGL')['length_of_stay'].describe().round(4)


# In[19]:


# Here we train the model
X_lang_train = X_train.join(X_lang, how='inner')
X_lang_test = X_test.join(X_lang, how='inner')
lang_model = XGBClassifier()
lang_model.fit(X_lang_train, y_train)
y_pred_lang = lang_model.predict(X_lang_test)
#
print('\n', "ROC_AUC Score with Gender Included:", roc_auc_score(y_test, y_pred_lang) )


# By comparing the results with and without the sensitivie attribute ...
# > to do: add explanation

# In[20]:


lang_scores = tutorial_helpers.get_aif360_measures_df(X_lang_test, y_test, y_pred_lang, sensitive_attributes=['LANG_ENGL'])
lang_scores.round(4).head(2)


# In[21]:


lang_ko_scores = tutorial_helpers.get_aif360_measures_df(X_lang_test, y_test, baseline_y_pred, sensitive_attributes=['LANG_ENGL']) 
lang_ko_scores.round(4).head(2)


# ### Comparing All Four Models Against Each Other

# In[22]:


full_comparison = comparison.merge(lang_scores.rename(columns={'value':'lang_score'})
                            ).merge(lang_ko_scores.rename(columns={'value':'lang_score (feature removed)'})
                            )
full_comparison.round(4)


# ## Part 5: Comparison to FairLearn <a class="anchor" id="part5"></a>
# 
# > to do: "FairLearn is another library..."
# 
# > to do: add comparison table
# 
# > to do: add peformance functions like roc_auc_score_group_summary from AIF360 to process above

# In[23]:


print("Selection rate", 
      selection_rate(y_test, y_pred_lang) )
print("Demographic parity difference", 
      demographic_parity_difference(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))
print("Demographic parity ratio", 
      demographic_parity_ratio(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))

print("------")
y_prob_lang = lang_model.predict_proba(X_lang_test)[:, 1]
print("Overall AUC", roc_auc_score(y_test, y_prob_lang) )
print("AUC difference", roc_auc_score_group_summary(y_test, y_prob_lang, sensitive_features=X_lang_test['LANG_ENGL']))


# ### Balanced Error Rate Difference
# Similar to the Equal Opportunity Difference measured by AIF360, the Balanced Error Rate Difference offered by FairLearn calculates the difference in accuracy score between the 

# In[24]:



print("-----")
print("Balanced error rate difference",
        balanced_accuracy_score_group_summary(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))
print("Equalized odds difference",
      equalized_odds_difference(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))
      


# # Summary
# > to do: "In this tutorial..."

# # References 
# 
# MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available from: http://www.nature.com/articles/sdata201635
# 
# <a id="hcup"></a>
# HCUP https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp

# In[ ]:





# In[ ]:





# # TEST CELLS (TO BE REMOVED BEFORE TUTORIAL)

# ## AIF360

# ### Effect of Caucasian Ethnicity

# In[82]:


eth_cols = [c for c in df.columns if c.startswith("ETHNICITY_")]
cauc_cols = [c for c in df.columns if c.startswith("ETHNICITY_WHITE")]
cauc_cols


# In[83]:


X_eth =  df.loc[:,eth_cols]
X_eth['caucasian'] = 0
X_eth.loc[X_eth[cauc_cols].eq(1).any(axis=1), 'caucasian'] = 1
X_eth = X_eth.drop(eth_cols, axis=1).fillna(0)
X_eth['caucasian'].describe()


# In[84]:


X_eth.join(df['length_of_stay']).groupby('caucasian')['length_of_stay'].describe()


# In[85]:


X_eth_train = X_train.join(X_eth, how='inner')
X_eth_test = X_test.join(X_eth, how='inner')
print(X_train.shape, X_eth_train.shape, X_test.shape, X_eth_test.shape)


# In[86]:


eth_model = XGBClassifier()
eth_model.fit(X_eth_train, y_train)
eth_y_pred = eth_model.predict(X_eth_test) 


# In[87]:


(tutorial_helpers.get_aif360_measures_df(X_eth_test, y_test, y_pred, sensitive_attributes=['caucasian']) ).round(4)


# In[88]:


eth_scores = tutorial_helpers.get_aif360_measures_df(X_eth_test, y_test, eth_y_pred, sensitive_attributes=['caucasian'])
eth_scores.round(4)


# In[ ]:




