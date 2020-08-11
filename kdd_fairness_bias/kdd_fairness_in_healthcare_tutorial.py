#!/usr/bin/env python
# coding: utf-8

# # Tutorial of Fairness Metrics for Healthcare
# 
# ### Overview
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
# 
# The tutorial also uses data from the MIMIC III Critical Care database, a freely accessible source of Electronic Health Records from Beth Israel Deaconess Medical Center in Boston. To download the MIMIC III data, please use this link: [Access to MIMIC III](https://mimic.physionet.org/gettingstarted/access/). Please save the data with the default directory name ("MIMIC"). No further action is required beyond remembering the download location: you do not need to unzip any files.

# ## Part 0: Background 
# SECTIONS TO BE INCLUDED:
# * what is fairness
# * metrics for fairness
# * list of measures that will be included in this notebook

# ## Part 1: Model Setup <a class="anchor" id="part1"></a>
# 
# This section introduces and loads the data subset that will be used in this tutorial. Then it generates a simple baseline model to be used throughout the tutorial.

# In[1]:


# Standard Libraries
from IPython.display import Image
import numpy as np
import os
import pandas as pd
import sys

# Prediction Libraries
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

# Helpers from local folder
import tutorial_helpers


# ### MIMIC III Data Subset
# As mentioned aboce, the MIMIC-III data download contains a folder of zipped_files. The tutorial code will automatically unzip and format all necessary data for these experiments, saving the formatted data in the tutorial folder. Simply enter the correct path of the MIMIC folder in the following cell to enable this feature. Your path should end with the directory "MIMIC".
# 
# Example: path_to_mimic_data_folder = "~/data/MIMIC"

# In[2]:


# path_to_mimic_data_folder = "[path to your downloaded data folder]"
path_to_mimic_data_folder = "~/data/MIMIC"


# ### Data Subset
# Example models in this notebook use data from all years of the MIMIC-III dataset for patients aged 65 and older. Data are imported at the encounter level with all additional patient identification dropped. All models include an "AGE" feature, simplified to 5-year bins, as well as boolean diagnosis and procedure features categorized through the Clinical Classifications Software system ([HCUP](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)). All features other than age are one-hot encoded and prefixed with their variable type (e.g. "GENDER_", "ETHNICITY_").  

# In[3]:


df = tutorial_helpers.load_example_data(path_to_mimic_data_folder) # note: ADMIT_ID has been masked
df.head()


# ### Baseline Length of Stay Model
# Example models in this tutorial predict the length of time spent in the ICU, a.k.a. the "Length of Stay" (LOS). The baseline model will use only the patient's age, their diagnosis, and the use of medical procedures during their stay to predict this value. 
# 
# Two target variables will be used in the following experiments: 'length_of_stay' and 'los_binary'. For this dataset, length_of_stay is, of course, the true value of the length of the patient's stay in days. The los_binary variable is a binary variable indicating whether the admission resulted in a length of stay either < or >= the mean. We will generate variable below, and then generate our baseline model.

# In[4]:


mean_val=df['length_of_stay'].mean()
df['los_binary'] = df['length_of_stay'].apply(lambda x: 0 if x < mean_val else 1)
df[['length_of_stay', 'los_binary']].describe().round(4)


# In[5]:


# Subset and Split Data
X = df.loc[:,['ADMIT_ID']+[c for c in df.columns if (c.startswith('AGE') or c.startswith('DIAGNOSIS_') or c.startswith('PROCEDURE_'))]]
y = df.loc[:, ['los_binary']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Train Model
baseline_model = XGBClassifier()
baseline_model.fit(X_train, y_train)
baseline_y_pred = baseline_model.predict(X_test)
baseline_y_prob = baseline_model.predict_proba(X_test)[:, 1]
#
print('\n', "Baseline ROC_AUC Score:", roc_auc_score(y_test, baseline_y_prob) )


# ## Part 2: Testing Gender as a Sensitive Attribute <a class="anchor" id="part2"></a>
# Our first experiment will test the effect of including the sensitive attribute 'GENDER_M'. This attribute is encoded in our data as a boolean attribute, where 0=female and 1=male, since males are assumed to be the privileged group. For the purposes of this experiment all other senstitive attributes and potential proxies will be dropped, such that only gender, age, diangosis, and procedure codes will be used to make the prediction.
# 
# First we will examine fairness measurements for a version of this model that includes gender as a feature, before comparing them to similar measurements for the baseline (without gender). We will see that while some measures can be used to analyze a model in isolation, others require comparison against other models.

# In[6]:


df.groupby('GENDER_M')['length_of_stay'].describe().round(4)


# In[7]:


# Update Split Data to Include Gender as a Feature
X_train_gender = X_train.join(df[['GENDER_M']], how='inner')
X_test_gender = X_test.join(df[['GENDER_M']], how='inner')
# Train New Model with Gender Feature
gender_model = XGBClassifier()
gender_model.fit(X_train_gender, y_train)
y_pred_gender = gender_model.predict(X_test_gender)

#
print('\n', "ROC_AUC Score with Gender Included:", roc_auc_score(y_test, y_pred_gender) )


# ### Measuring Fairness via AIF360
# 
# AIF360 requires the sensitive attribute to be in the same dataframe (or 2-D array) as the target variable (both the ground truth and the prediction), so we add that here.
# 

# In[8]:


y_test_aif = pd.concat([X_test_gender['GENDER_M'], y_test], axis=1).set_index('GENDER_M')
y_pred_aif = pd.concat([X_test_gender['GENDER_M'].reset_index(drop=True), pd.Series(y_pred_gender)], axis=1).set_index('GENDER_M')
y_pred_aif.columns = y_test_aif.columns


# ### Prediction Rates
# The base rate is the average value of the ground truth (optionally weighted). It provides useful context, although it is not technically a measure of fairness. 
# > $base\_rate = \sum_{i=0}^N(y_i)/N$
# 
# The Selection Rate is the average value of the predicted (ŷ).
# > $selection\_rate = \sum_{i=0}^N(ŷ_i)/N$

# In[9]:


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
# > $statistical\_parity\_difference = selection\_rate(ŷ_{unprivileged}) - selection\_rate(ŷ_{privileged}) $

# In[10]:


model_scores.loc[1] = ['disparate_impact_ratio', disparate_impact_ratio(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.loc[2] = ['statistical_parity_difference', statistical_parity_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.tail(2)


# ### Measures of Equal Odds
# Average Odds Difference measures the average of the difference in FPR and TPR for the unprivileged and privileged groups.
# > $ average\_odds\_difference = \dfrac{(FPR_{unprivileged} - FPR_{privileged})
#         + (TPR_{unprivileged} - TPR_{privileged})}{2}$
# 
# Average Odds Error is the average of the absolute difference in FPR and TPR for the unprivileged and privileged groups.
# > $average\_odds\_error = \dfrac{|FPR_{unprivileged} - FPR_{privileged}|
#         + |TPR_{unprivileged} - TPR_{privileged}|}{2}$
#         
# Equal Opportunity Difference is the difference in recall scores (TPR) between the unprivileged and privileged groups. A difference of 0 indicates that the model is fair.
# > $equal\_opportunity\_difference =  Recall(ŷ_{unprivileged}) - Recall(ŷ_{privileged})$
# 

# In[11]:


model_scores.loc[3] = ['average_odds_difference', average_odds_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.loc[4] = ['average_odds_error', average_odds_error(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.loc[5] = ['equal_opportunity_difference', equal_opportunity_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')]
model_scores.tail(3)


# ## Custom Measures of Disparate Performance
# Both of the libraries we will explore have added features to calculate the between-group difference in performance. Below we demonstrate a 

# In[48]:


performance_function = roc_auc_score
y_prob_gender = gender_model.predict_proba(X_test_gender)[:, 1]
model_scores.loc[6] = ['Between-Group AUC Difference', difference(roc_auc_score, y_test_aif, y_prob_gender, prot_attr='GENDER_M', priv_group=1)]
model_scores.tail(1)


# ### Measures Of Individual Fairness
# Consistency scores measure the similarity between a given prediction and the predictions of "like" individuals. In AIF360, the consistency score is calculated as the compliment of the mean distance to the score of the mean nearest neighbhor, using Scikit's Nearest Neighbors algorithm (default 5 neighbors determined by BallTree algorithm).
# > $ consistency\_score = 1 - |mean_{distance}(mean({nearest\ neighbor}) )| $
# 
# #### The Generalized Entropy Index and Related Measures
# The Generalized Entropy (GE) Index is...
# > $ GE =  \mathcal{E}(\alpha) = \begin{cases}
#             \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
#             \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
#             -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
#         \end{cases}
#         $
# 
# Generalized Entropy Error = Calculates the GE of the set of errors, i.e. 1 + (ŷ == pos_label) - (y == pos_label) 
# > $ GE(Error) = b_i = \hat{y}_i - y_i + 1 $
# 
# Between Group Generalized Entropy Error = Calculates the GE of the set of mean errors for the two groups (privileged error & unprivileged error), weighted by the number of predictions in each group
# > $ GE(Error_{group}) =  GE( [N_{unprivileged}*mean(Error_{unprivileged}), N_{privileged}*mean(Error_{privileged})] ) $

# In[13]:


model_scores.loc[7] = ['consistency_score', consistency_score(X_test_gender, y_pred_gender)]
model_scores.loc[8] = ['generalized_entropy_error', generalized_entropy_error(y_test['los_binary'], y_pred_gender)]
model_scores.loc[9] = ['between_group_generalized_entropy_error', 
                            between_group_generalized_entropy_error(y_test_aif, y_pred_aif, prot_attr=['GENDER_M'])]
model_scores.tail(3) 


# ## Part 3: Comparing Against a Second Model - Evaluating Unawareness
# <a class="anchor" id="part3"></a>
# 
# To demonstrate the change in model scores relative to the use of a sensitive attribute, we will compare the above scores to those of our baseline model. Although the GENDER_M feature is not included in our baseline, since we attached it to the y_test_aif dataframe above we can still evaluate it's bias relative to GENDER_M. As shown below, there is no significant difference in the scores of these two models. Therefore, the inclusion of GENDER_M as a feature does not contribute to gender bias for these models.
# 
# Note: Since we have already discussed the individual measures, a helper function will be used to save space.

# In[49]:


# Measure Values for Baseline Model, Relative to Patient Gender
new_scores = tutorial_helpers.get_aif360_measures_df(X_test_gender, y_test, baseline_y_pred, baseline_y_prob, sensitive_attributes=['GENDER_M'])


# In[50]:


comparison = model_scores.rename(columns={'value':'gender_score'}
                                ).merge(new_scores.rename(columns={'value':'gender_score (feature removed)'}))
comparison.round(4)


# ## Part 4: Testing Other Sensitive Attributes
# 
# Our next experiment will test the presence of bias relative to a patient\'s language, assuming that there is a bias toward individuals who speak English. As above, we will add a boolean 'LANGUAGE_ENGL' feature to the baseline data.

# In[25]:


# Update Split Data to Include Language as a Feature
lang_cols = [c for c in df.columns if c.startswith("LANGUAGE_")]
eng_cols = ['LANGUAGE_ENGL']
X_lang =  df.loc[:,lang_cols]
X_lang['LANG_ENGL'] = 0
X_lang.loc[X_lang[eng_cols].eq(1).any(axis=1), 'LANG_ENGL'] = 1
X_lang = X_lang.drop(lang_cols, axis=1).fillna(0)
X_lang.join(df['length_of_stay']).groupby('LANG_ENGL')['length_of_stay'].describe().round(4)


# In[28]:


# Train New Model with Language Feature
X_lang_train = X_train.join(X_lang, how='inner')
X_lang_test = X_test.join(X_lang, how='inner')
lang_model = XGBClassifier()
lang_model.fit(X_lang_train, y_train)
y_pred_lang = lang_model.predict(X_lang_test)
y_prob_lang = lang_model.predict_proba(X_lang_test)
print('\n', "ROC_AUC Score with Gender Included:", roc_auc_score(y_test, y_pred_lang) )


# Again, by comparing the results with and without the sensitivie attribute we can better demonstrate the effect that the attribute has on the fairness of the model. In this example we see

# In[46]:


# Measure Values for Language-Inclusive Model, Relative to Patient Language
print("Measure values with feature included:")
lang_scores = tutorial_helpers.get_aif360_measures_df(X_lang_test, y_test, y_pred_lang, y_prob_lang, sensitive_attributes=['LANG_ENGL'])
print(lang_scores.round(4).head(3))
# Measure Values for Baseline Model, Relative to Patient Language
print("\n", "Measure values with feature removed:")
lang_ko_scores = tutorial_helpers.get_aif360_measures_df(X_lang_test, y_test, baseline_y_pred, baseline_y_prob, sensitive_attributes=['LANG_ENGL']) 
print(lang_ko_scores.round(4).head(3))


# ### Comparing All Four Models Against Each Other
# As shown below, exclusion of the LANG_ENGL feature has a more significant impact on the fairness of the model than does exclusion of GENDER_M (relative to their specific biases). Moreover, using the 80/20 rule we can see that inclusion of LANG_ENGL leads to what can be considered a "significant" bias, as shown by the Disparate Impact Ratio. In this case, predictions for those individuals who do not speak English are significantly more likely to be above the mean, even though this difference is not [currently] reflected in the ground truth.
# 
# > to do: validate this conclusion after fixing the issue with LOS <- currently the average LOS in the dataset is still higher than expected

# In[51]:


full_comparison = comparison.merge(lang_scores.rename(columns={'value':'lang_score'})
                            ).merge(lang_ko_scores.rename(columns={'value':'lang_score (feature removed)'})
                            )
full_comparison.round(4)


# ## Part 5: Comparison to FairLearn <a class="anchor" id="part5"></a>
# 
# Next, some of the same metrics will be demonstrated using Microsoft's FairLearn library. Although both APIs are similar and the measures built into FairLearn are not as comprehensive as those of AIF360, some users may find FairLearn's documentation style to be more accessible. A table comparing the measures available in each library is shown below. 

# In[35]:


Image(url="img/library_measure_comparison.png", width=500)


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
print("Between-Group AUC Difference", roc_auc_score_group_summary(y_test, y_prob_lang, sensitive_features=X_lang_test['LANG_ENGL']))


# ### Balanced Error Rate Difference
# Similar to the Equal Opportunity Difference measured by AIF360, the Balanced Error Rate Difference offered by FairLearn calculates the difference in accuracy score between the unprivileged and privileged group.
# 
# > to do: expand upon this...

# In[24]:



print("-----")
print("Balanced error rate difference",
        balanced_accuracy_score_group_summary(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))
print("Equalized odds difference",
        equalized_odds_difference(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))
      


# ## Summary
# This tutorial introduced multiple measures of ML fairness in the context of a healthcare model using the AIF360 and FairLearn Python libraries. A subset of the MIMIC-III database was used to generate a series of simple Length of Stay (LOS) models. It was shown that while the inclusion of a sensitive feature can significantly affect a model's bias as it relates to that feature, this is not always the case. 

# # References 
# [AIF360 Reference](http://aif360.mybluemix.net/) 
# 
# [HCUP Reference](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp
# 
# [FairLearn Reference](https://fairlearn.github.io/).
# 
# MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available from: http://www.nature.com/articles/sdata201635
# 
