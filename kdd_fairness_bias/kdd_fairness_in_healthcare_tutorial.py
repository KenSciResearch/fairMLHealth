#!/usr/bin/env python
# coding: utf-8

# # Measuring Fairness in Healthcare ML for Scikit-Compatible Models
# 
# ### Overview
# This tutorial introduces methods and libraries for measuring fairness and bias in machine learning models as as they relate to problems in healthcare. Through the tutorial you will first learn some basic background about fairness and bias in machine learning. You will then generate a simple baseline model predicting Length of Stay (LOS) using data from the [MIMIC-III database](https://mimic.physionet.org/gettingstarted/access/), which you will use as an example to understand the most prominent fairness measures. You will also gain familiarity with the Scikit-Learn-compatible tools available in [AIF360](http://aif360.mybluemix.net/) and [FairLearn](https://fairlearn.github.io/), two of the most comprehensive and flexible Python libraries for measuring and addressing bias in machine learning models.
#   
# ### Tutorial Contents
# [Part 0](#part0) - Metrics of Fairness
# 
# [Part 1](#part1) - Model Setup
# 
# [Part 2](#part2) - Metrics of Fairness in AIF360
# 
# [Part 3](#part3) - Comparing Against a Second Model: Evaluating Unawarenes
# 
# [Part 4](#part4) - Testing Other Sensitive Attributes
# 
# [Part 5](#part5) - Comparison to FairLearn
# 
# ### Tutorial Requirements
# This tutorial assumes basic knowledge of machine learning implementation in Python. Before starting, please install [AIF360](http://aif360.mybluemix.net/) and [FairLearn](https://fairlearn.github.io/). Also, ensure that you have installed the Scipy, Pandas, Numpy, Scikit, and XGBOOST libraries.
# 
# The tutorial also uses data from the MIMIC III Critical Care database, a freely accessible source of Electronic Health Records from Beth Israel Deaconess Medical Center in Boston. To download the MIMIC III data, please use this link: [Access to MIMIC III](https://mimic.physionet.org/gettingstarted/access/). Please save the data with the default directory name ("MIMIC"). No further action is required beyond remembering the download location: you do not need to unzip any files.
# 

# # Part 0 - Metrics of Fairness <a class="anchor" id="part0"></a>
# ## What is a "Fair" Model
# In the context of this tutorial, a "fair" model is one that does not impose discrimination - the unjustified differential treatment of individuals based on their demographic status.[Romei and Ruggieri (2014)](#romei2014_ref)  There are six common metrics for determining whether a given machine learning model meets this definition: equal treatment ("unawareness"), demographic parity, equalized odds, predictive parity, individual fairness, and counterfactual fairness. Basic definitions for each of these metrics are provided for you in the [Quick Reference](#metric_quickref) below. 
# 
# Each metric can be assesed by one of several different measures, the choice of which depends upon the analysis in question. This tutorial will guide you through many of these measures by introducing you to the scikit-learn compatible APIs available in the [AIF360](http://aif360.mybluemix.net/) and [FairLearn](https://fairlearn.github.io/) libraries. 
# 
# ## Disparate Treatment vs Disparate Impact <a id="disparateimpact"></a>
# Discussions of discrimination often evoke direct, typically deliberate action to disadvantage one race, religion, or ethnicity. Such explicit action is often referred to as *disparate treatment*. However, a more insidious form of discrimination emerges from seemingly unbiased practices that result (often unwittingly) in the unfair treatment of a socially disadvantaged group. In machine learning models, such *disparate impact* can emerge, for example, from unconscious collection biases or undetected historical biases in the data, leading either to immediate disparity in model predictions or to a longer-term discrepancy through unconscious human biases in its application. This tutorial focuses on metrics to determine the former. However, if you are interested in the latter, we reccomend looking at the [Fairness Indicators](https://www.tensorflow.org/tfx/fairness_indicators) suite released by a team at Google.
# 

# ##  Fairness Metrics Quick Reference <a id="metric_quickref"></a>
# | Metric | Defintion | Weakness | References |
# |------|------|------|------|
# | Unawareness | A model is unaware if the sensitive attribute is not used. | Removal of a sensitive attribute may be ineffectual due to the presence of proxy features highly correlated with the sensitive attribute. | [Zemel *et al* (2013)](#zemel2013_ref), [Barocas and Selbst (2016)](#barocas2016_ref) |
# | Demographic Parity | A model has demographic parity if the predicted positve rates (selection rates) are approximately the same for all sensitive attribute groups.<br> P(ŷ =1 \| unprivileged) / P(ŷ =1 \| privileged) | Historical biases present in the data are not addressed and may still bias the model. | [Zafar *et al* (2017)](#zafar2017_ref) |
# | Equalized Odds | Odds are equalized if P(+) is approximately the same for all sensitive attribute groups.<br>  **Equal Opportunity** is a special case of equalized odds specifying that P(+\|y=1) is approximately the same across groups. | Historical biases present in the data  are not addressed and may still bias the model. | [Hardt *et al* (2016)](#hardt2016_ref) |
# | Predictive Parity | Predictive parity exists where the Positive Predictive Value and Negative Predictive Value are each approximately the same for all sensitive attribute groups. | Historical biases present in the data are not addressed and may still bias the model.  | [Zafar *et al* (2017)](#zafar2017_ref) |
# | Individual Fairness | Individual fairness exists if "similar" individuals (ignoring the sensitive attribute) are likely to have similar predictions. | The appropriate metric for similarity may be ambiguous. |[Dwork (2012)](#dwork2012_ref), [Kim *et al* (2018)](#kim2018_ref) |
# | Counterfactual Fairness** | Counterfactual fairness exists where counterfactual replacement of the sensitive attribute does not significantly alter predictive performance. This counterfactual change must be propogated to correlated variables. | It may be intractable to develop a counterfactual model.  | [Russell *et al* (2017)](#russell2017_ref) |
# 
# ## Measures Covered in this Tutorial
# | Metric | Measures Covered |
# |----|----|
# |Unawareness| *Measured through model comparison using other metrics of fairness.* |
# |Demographic Parity| Disparate Impact Ratio, Statistical Parity Difference|
# |Equalized Odds| Average Odds Difference, Equal Odds Ratio, Equal Odds Error, Equalized Odds Difference, Equalized Odds Ratio|
# |Predictive Parity| (Positive) Predictive Parity Difference|
# |Individual Fairness| Consistency Score, Generalized Entropy Error, Between-Group Generalized Entropy Error|
# |Counterfactual Fairness**| *Counterfactual fairness is not covered in this tutorial.* |
# 
# ** Note that this tutorial will not elaborate the details of counterfactual fairness since the libraries used do not have built-in functionality for it.
# 

# # Part 1 - Model Setup <a class="anchor" id="part1"></a>
# 
# This section introduces and loads the data subset that will be used in this tutorial. We will use it to generate a simple baseline model that will be used throughout the tutorial.

# In[125]:


# Standard Libraries
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
import sys

# Prediction Libraries
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# Metrics
import sklearn.metrics as sk_metrics
from aif360.sklearn.metrics import *
from fairlearn.metrics import (
    selection_rate as fairlearn_selection_rate, 
    demographic_parity_difference, demographic_parity_ratio,
    balanced_accuracy_score_group_summary, roc_auc_score_group_summary,
    equalized_odds_difference, equalized_odds_ratio,
    difference_from_summary)

# Helpers from local folder
import tutorial_helpers

# Remove limit to the number of columns and column widths displayed by pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 0) # adjust column width as needed


# ## Loading MIMIC III Data
# As mentioned above, the MIMIC-III data download contains a folder of zipped_files. The tutorial code in the [Data Subset](#datasubset) section below will automatically unzip and format all necessary data for these experiments, saving the formatted data in the same MIMIC folder. Simply enter the correct path of the MIMIC folder in the following cell to enable this feature. Your path should end with the directory "MIMIC".
# 
# Example: path_to_mimic_data_folder = "~/data/MIMIC"

# In[32]:


# path_to_mimic_data_folder = "[path to your downloaded data folder]"
path_to_mimic_data_folder = "~/data/MIMIC"


# ## Data Subset <a id="datasubset"></a>
# Example models in this notebook use data from all years of the MIMIC-III dataset for patients aged 65 and older. Data are imported at the encounter level with all additional patient identification dropped. All models include an "AGE" feature, simplified to 5-year bins, as well as boolean diagnosis and procedure features categorized through the Clinical Classifications Software system ([HCUP](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)). All features other than age are one-hot encoded and prefixed with their variable type (e.g. "GENDER_", "ETHNICITY_").  

# In[158]:


df = tutorial_helpers.load_example_data(path_to_mimic_data_folder) 
tutorial_helpers.print_feature_table(df)
display(df.head())


# ## Baseline Length of Stay Model
# The example models in this tutorial predict the total length of time spent in the Intensive Care Unity (ICU) for a given hospital admission, a.k.a. the "Length of Stay" (LOS). The baseline model will use only the patient's age, their diagnosis, and the use of medical procedures during their stay to predict this value. 
# 
# Two target variables will be used in the following experiments: 'length_of_stay' and 'long_los'. The length_of_stay target contains the true value in days for the patient's stay in ICU. The long_los target is a binary variable indicating whether or not the length_of_stay for a given admission is greater than the mean. Observations have been dropped where LOS values are extremely long (30 days or greater) or negative (indicating that the patient was deceased upon arrival at the ICU). We will generate variable below, then generate our baseline model.

# In[87]:


# Generate a binary target flagging whether an observation's length_of_stay value is above or below the mean. 
mean_val=df['length_of_stay'].mean()
df['long_los'] = df['length_of_stay'].apply(lambda x: 1 if x > mean_val else 0)
display(df[['length_of_stay', 'long_los']].describe().transpose().round(4))

# Display LOS distributions
ax = df['length_of_stay'].plot(kind='kde', title="Probability Density of Length of Stay")
ax.set_xlabel("Length of Stay in Days")
plt.show()


# In[41]:


# Subset and Split Data
X = df.loc[:,[c for c in df.columns if (c.startswith('AGE') or c.startswith('DIAGNOSIS_') or c.startswith('PROCEDURE_'))]]
y = df.loc[:, ['long_los']]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)
xgb_params = {'colsample_bytree': 1.0, 'gamma': 2, 'learning_rate': 0.05, 'max_depth': 5, 
                'min_child_weight': 1,  'n_estimators': 600, 'subsample': 0.6}

# Train Model
baseline_model = XGBClassifier(**xgb_params)
baseline_model.fit(X_train, y_train.iloc[:,0])
baseline_y_pred = baseline_model.predict(X_test)
baseline_y_prob = baseline_model.predict_proba(X_test)

print("\n", "Prediction Scores for Baseline Model:", "\n", 
      sk_metrics.classification_report(y_test.iloc[:,0], baseline_y_pred, target_names=['LOS <= mean', 'LOS > mean']) )


# # Part 2 - Testing Gender as a Sensitive Attribute <a class="anchor" id="part2"></a>
# Our first experiment will test the effect of including the sensitive attribute 'GENDER_M'. This attribute is encoded in our data as a boolean attribute, where 0=female and 1=male, since males are assumed to be the privileged group. For the purposes of this experiment all other senstitive attributes and potential proxies will be dropped, such that only gender, age, diangosis, and procedure codes will be used to make the prediction.
# 
# First we will examine fairness measurements for a version of this model that includes gender as a feature, before comparing them to similar measurements for the baseline (without gender). We will see that while some measures can be used to analyze a model in isolation, others (e.g. unawareness) require comparison against other models to clearly demonstrate a model's fairness/unfairness.

# In[82]:


# Show LOS Distribtuion Relative to GENDER_M
is_male = df['GENDER_M'].eq(1)
display(df.groupby('GENDER_M')['length_of_stay'].describe().round(4))
ax_g = df.groupby('GENDER_M')['length_of_stay'].plot(kind='kde', title="Probability Density of Length of Stay by Gender")
plt.show()

# Display Results of Independent T-Test
deg_f = 2*(len(df)) - 2
ttest = stats.ttest_ind(df.loc[is_male,'length_of_stay'],df.loc[~is_male,'length_of_stay'])
print("\n", "Results of Independent T-Test", "\n",
      "Degrees of Freedom:", deg_f, "\n",
      "T Value:", ttest[0], "\n",
      "P Value:", ttest[1]
     )


# In[43]:


# Update Split Data to Include Gender as a Feature
X_train_gender = X_train.join(df[['GENDER_M']], how='inner')
X_test_gender = X_test.join(df[['GENDER_M']], how='inner')

# Train New Model with Gender Feature
gender_model = XGBClassifier()
gender_model.fit(X_train_gender, y_train.iloc[:,0])
y_pred_gender = gender_model.predict(X_test_gender)
y_prob_gender = gender_model.predict_proba(X_test_gender)

print("\n", "Prediction Scores for Model with Gender Included:", "\n", 
      sk_metrics.classification_report(y_test.iloc[:,0], y_pred_gender, target_names=['LOS <= mean', 'LOS > mean']) )


# ## Measuring Fairness via AIF360
# 
# **Formatting Note:** AIF360 requires the sensitive attribute(s) to be the index of the dataframes (or 2-D arrays) containing the ground truth and the prediction targets. For our walkthrough of the different measures we will add that formatting in the cell below. Note that Dataframes that have been formatted for the AIF360 Scikit API will be given the suffix "_aif".
# 

# In[44]:


# Generate target dataframes specifically formatted as inputs for AIF360
y_test_aif = pd.concat([X_test_gender['GENDER_M'], y_test], axis=1).set_index('GENDER_M')
y_pred_aif = pd.concat([X_test_gender['GENDER_M'].reset_index(drop=True), pd.Series(y_pred_gender)], axis=1).set_index('GENDER_M')
y_pred_aif.columns = y_test_aif.columns


# ### Prediction Rates
# Since the effect of Unawareness is difficult to measure without comparison against another model, we will begin our exploration of AIF360 by demonstrating some common terminology. For the curious, Unawareness will be dissused in [Part 3](#part3). 
# 
# The *Base Rate* returns the average value of the ground truth (optionally weighted). It provides useful context, although it is not technically a measure of fairness. 
# > $base\_rate = \sum_{i=0}^N(y_i)/N$
# 
# The *Selection Rate* returns the average predicted value. For a binary prediction problem it equates to the probability of prediction for the positive class ("the probability of selection").
# > $selection\_rate = \sum_{i=0}^N(ŷ_i)/N$

# In[45]:


# Demonstrate that the Base Rate and Selection Rate return mean values
print("base_rate = ", base_rate(y_test_aif),
      "\n\tP(y_+) =", y_test_aif['long_los'].mean()
     )
print('overall selection_rate =', selection_rate(y_test_aif, y_pred_aif),
      "\n\tP(ŷ_+) =", y_pred_aif['long_los'].mean()
     )

# Display the group-specific selection rates
females = (y_pred_aif.index.get_level_values('GENDER_M') == 0)
print()
print('Selection rate for the unprivileged group =', y_pred_aif.loc[females, 'long_los'].mean())
print('Selection rate for the privileged group =', y_pred_aif.loc[~females, 'long_los'].mean())


# ### Measures of Demographic Parity
# A model has demographic parity if the predicted positve rates (selection rates) are approximately the same for all sensitive attribute groups. Two common measures for demographic parity are the Statistical Parity Difference and the Disparate Impact Ratio.
# 
# The *Statistical Parity Difference* is the difference in the probability of prediction between the two groups. A difference of 0 indicates that the model is fair relative to the sensitive attribute (it favors neither the privileged nor the unprivileged group).
# > $statistical\_parity\_difference = P(ŷ =1 | unprivileged) - P(ŷ =1 | privileged) $
# 
# The *Disparate Impact Ratio* is the ratio between the probability of positive prediction for the unprivileged group and the probability of positive prediction for the privileged group. A ratio of 1 indicates that the model is fair relative to the sensitive attribute (it favors neither the privileged nor the unprivileged group).
# > $disparate\_impact\_ratio = \dfrac{P(ŷ =1 | unprivileged)}{P(ŷ =1 | privileged)} = \dfrac{selection\_rate(ŷ_{unprivileged})}{selection\_rate(ŷ_{privileged})}$
# 

# In[248]:


# Display the Disparate Impact Ratio and Statistical Parity Difference
print('disparate_impact_ratio \n', '\t', 
      disparate_impact_ratio(y_test_aif, y_pred_aif, prot_attr='GENDER_M')
     )
print('statistical_parity_difference \n', '\t', 
      statistical_parity_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')
     )


# ### Measures of Equal Odds
# Odds are equalized if P(+) is approximately the same for all sensitive attribute groups.
# 
# *Average Odds Difference* measures the average of the difference in False Positive Rate (FPR) and True Positive Rate (TPR) for the unprivileged and privileged groups. A difference of 1 indicates that the model is fair relative to the sensitive attribute.
# > $ average\_odds\_difference = \dfrac{(FPR_{unprivileged} - FPR_{privileged})
#         + (TPR_{unprivileged} - TPR_{privileged})}{2}$
# 
# *Average Odds Error* is the average of the absolute difference in FPR and TPR for the unprivileged and privileged groups. A difference of 1 indicates that the model is fair relative to the sensitive attribute.
# > $average\_odds\_error = \dfrac{|FPR_{unprivileged} - FPR_{privileged}|
#         + |TPR_{unprivileged} - TPR_{privileged}|}{2}$
#         
# *Equal Opportunity Difference* is the difference in recall scores (TPR) between the unprivileged and privileged groups. A difference of 0 indicates that the model is fair relative to the sensitive attribute.
# > $equal\_opportunity\_difference =  Recall(ŷ_{unprivileged}) - Recall(ŷ_{privileged})$
# 

# In[15]:


# Display measures of Equal Odds
print('average_odds_difference', '\t', 
       average_odds_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M'))
print('average_odds_error', '\t', 
       average_odds_error(y_test_aif, y_pred_aif, prot_attr='GENDER_M')
     )
print('equal_opportunity_difference', '\t', 
       equal_opportunity_difference(y_test_aif, y_pred_aif, prot_attr='GENDER_M')
     )


# ### Predictive Rate Parity and Disparate Performance
# Predictive parity exists where the Positive Predictive Value and Negative Predictive Value are each approximately the same for all sensitive attribute groups.
# 
# The *Positive Predictive Paritiy Difference* is the difference in Positive Predictive Value (PPV), aka. the precision. 
# > $positive\_predictive\_parity\_difference =  precision(ŷ_{unprivileged}) - precision(ŷ_{privileged})$
# 
# <a id='aif_difference_func'></a>
# Although neither of the libraries in this tutorial have a specific function for measuring Predictive Parity, both libraries do include features for calculaing custom, between-group differences in performance. In AIF360, this is facilitated by a flexible "difference" method that accepts an arbitrary scoring function as an argument. AIF360 also contains a "ratio" function of similar purpose. Here we will use difference() and ratio() to calculate  positive predictive parity difference for the GENDER_M model, in addition to several different custom metrics.
# 

# In[252]:


# Display examples using the difference function
print('\n', 'Positive Predictive Parity Difference',
          difference(sk_metrics.precision_score, y_test_aif, y_pred_gender, prot_attr='GENDER_M', priv_group=1)
     )
print('\n', 'Between-Group AUC Difference',
          difference(sk_metrics.roc_auc_score, y_test_aif, y_prob_gender[:, 1], prot_attr='GENDER_M', priv_group=1)
     )
print('\n', 'Between-Group Balanced Accuracy Difference', 
          difference(sk_metrics.balanced_accuracy_score, y_test_aif, y_pred_gender, prot_attr='GENDER_M', priv_group=1)
     )

# Display examples using the ratio function
print('\n', 'Selection Rate Ratio (Disparate Impact Ratio)', 
        ratio(selection_rate, y_test_aif, y_pred_gender, prot_attr='GENDER_M', priv_group=1) 
     )
print('\n', 'Recall Ratio', 
        ratio(sk_metrics.recall_score, y_test_aif, y_pred_gender, prot_attr='GENDER_M', priv_group=1) 
     )


# ### Measures Of Individual Fairness
# Measures of individual fairness determine if "similar" individuals (ignoring the sensitive attribute) are likely to have similar predictions.
# 
# #### Consistency Scores
# Consistency scores measure the similarity between a given prediction and the predictions of "like" individuals. In AIF360, the consistency score is calculated as the compliment of the mean distance to the score of the mean nearest neighbhor, using Scikit's Nearest Neighbors algorithm (default 5 neighbors determined by BallTree algorithm). For this measure, values closer to zero indicate greater fairness, with increasing values indicating decreased fairness. However, since this measure is dimensionless, its utility is only realized in comparison to other scores from similar models.
# > $ consistency\_score = 1 - |mean_{distance}(mean({nearest\ neighbor}) )| $
# 
# #### The Generalized Entropy Index and Related Measures
# The *Generalized Entropy (GE) Index* was originally proposed as a metric for income inequality ([Shorrocks 1980](#shorrocks_ref)), but originated as a measure of redundancy in information theory. In 2018, [Speicher *et al*](#speicher2018_ref) proposed its use for Machine Learning models.
# > $ GE = \mathcal{E}(\alpha) = \begin{cases}
#             \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
#             \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
#             -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
#         \end{cases}
#         $
# 
# *Generalized Entropy Error* is the generalized entropy index of the prediction error. Like the consistency_score above, this measure is dimensionless and thus most useful in comparison between models. Values closer to zero indicate greater fairness and increasing values indicating decreased fairness. 
# > $ GE(Error) = GE(\hat{y}_i - y_i + 1) $
# 
# *Between Group Generalized Entropy Error* is the generalized entropy index for the weighted means of group-specific errors. Again this measure is dimensionless, with values closer to zero indicate greater fairness and increasing values indicating decreased fairness. 
# > $ GE(Error_{group}) =  GE( [N_{unprivileged}*mean(Error_{unprivileged}), N_{privileged}*mean(Error_{privileged})] ) $
# 

# In[17]:


print('\n', 'consistency_score',
      consistency_score(X_test_gender, y_pred_gender)
     )
print('\n', 'generalized_entropy_error', 
      generalized_entropy_error(y_test['long_los'], y_pred_gender)
     )
print('\n', 'between_group_generalized_entropy_error', 
      between_group_generalized_entropy_error(y_test_aif, y_pred_aif, prot_attr=['GENDER_M'])
     )


# ## Quick Reference of Fairness Measures
# | Metric | Measure | <div style="width:400px" align="center">Definition</div>  | Interpretation |
# |----|----|----|----|
# | **General Measures** | **Base Rate** | $\sum_{i=0}^N(y_i)/N$  |  |
# | | **Selection Rate** | $\sum_{i=0}^N(ŷ_i)/N$ |  |
# | **Group Fairness Measures** | **Demographic (Statistical) Parity Difference** | $P(ŷ =1 | unprivileged) - P(ŷ =1 | privileged) $| 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
# | | **Disparate Impact Ratio (Demographic Parity Ratio)** | $\dfrac{P(ŷ =1 | unprivileged)}{P(ŷ =1 | privileged)} = \dfrac{selection\_rate(ŷ_{unprivileged})}{selection\_rate(ŷ_{privileged})}$ | 1 indicates fairness <br>  < 1 favors privileged group <br>  > 1 favors unprivileged group |
# | | **Generalized Between-Group Predictive Disparity (eg. difference in ROC)** |  |
# | | **Average Odds Difference** | $\dfrac{(FPR_{unprivileged} - FPR_{privileged}) + (TPR_{unprivileged} - TPR_{privileged})}{2}$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
# | | **Average Odds Error** | $\dfrac{|FPR_{unprivileged} - FPR_{privileged}| + |TPR_{unprivileged} - TPR_{privileged}|}{2}$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
# | | **Equal Opportunity Difference** | $Recall(ŷ_{unprivileged}) - Recall(ŷ_{privileged})$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
# | | **Equalized Odds Difference** | $max( (FPR_{unprivileged} - FPR_{privileged}), (TPR_{unprivileged} - TPR_{privileged}) )$ | 0 indicates fairness <br> (-) favors privileged group <br> (+) favors unprivileged group |
# | | **Equalized Odds Ratio**** | $min( \dfrac{FPR_{smaller}}{FPR_{larger}}, \dfrac{TPR_{smaller}}{TPR_{larger}} )$ | 1 indicates fairness <br>  < 1 favors privileged group <br>  > 1 favors unprivileged group |
# | **Individual Fairness Measures** | **Consistency Score** | $1 - |mean_{distance}(mean({nearest\ neighbor}) )$ | 0 indicates fairness<br>(+) indicates unfairness |
# | | **Generalized Entropy Index** |  |
# | | **Between-Group Generalized Entropy Error** | $GE(\hat{y}_i - y_i + 1) $ | 0 indicates fairness<br>(+) indicates unfairness |
# | | **Generalized Entropy Error** | $GE( [N_{unprivileged}*mean(Error_{unprivileged}), N_{privileged}*mean(Error_{privileged})] ) $ | 0 indicates fairness<br>(+) indicates unfairness |

# # Part 3 - Comparing Against a Second Model <a class="anchor" id="part3"></a>
# ### Evaluating Unawareness 
# A model is unaware relative to a sensitive attribute so long as that attribute is not included in the model. 
# 
# To facilitate comparison between our GENDER_M-inclusive model and the baseline, this tutorial includes a helper function which returns all of the previously seen measures in a convenient pandas dataframe. This function will be used to save space for the remainder of the tutorial. 
# 
# Below we generate a table conatining fairness scores for our LOS models. The scores we just generated are constrasted against gender-relative scores for the baseline model, which importantly does not contain GENDER_M as an attribute. As the table shows, removal of the gender attribute produces little change in measure values. 

# In[64]:


from importlib import reload
reload(tutorial_helpers)
# Use Tutorial Helper Function to Generate Dataframe of Measure Values for Gender-Included Model, Relative to Patient Gender
gender_scores = tutorial_helpers.get_aif360_measures_df(X_test_gender, y_test, y_pred_gender, y_prob_gender, sensitive_attributes=['GENDER_M'])

# Use Tutorial Helper Function to Generate Dataframe of Measure Values for Baseline Model, Relative to Patient Gender
baseline_scores = tutorial_helpers.get_aif360_measures_df(X_test_gender, y_test, baseline_y_pred, baseline_y_prob, sensitive_attributes=['GENDER_M'])

# Merge Results to Compare Values
comparison = gender_scores.rename(columns={'Value':'Gender Fairness (GENDER_M Included)'}
                                ).merge(baseline_scores.rename(columns={'Value':'Gender Fairness, Baseline (GENDER_M Excluded)'}))
comparison


# ### Evaluating Significance: The Four-Fifths Rule 
# As shown in the table above, the baseline model that excluded the GENDER_M feature had slightly greater performance to the gender-included version by some measures (for example ROC). However, the baseline model is also measured to have slightly increased disparity by some metrics. For example, the Statistical Parity difference, Disparate Impact Ratio, and Average Odds Difference values are all slightly higher for the baseline model than for the GENDER_M-included model. How do we determine if this change is significant?
# 
# In 1978, the United States Equal Employment Opportunity Commission adopted the "four-fifths rule", a guideline stating that, "A selection rate for any race, sex, or ethnic group which is less than four-fifths (4/5) (or eighty percent) of the rate for the group with the highest rate will generally be regarded... as evidence of adverse impact." [EOC (1978)](#fourfifths_ref) This rubric has since been adopted for measures of fairness in machine learning as well, although the group with the "highest rate" is considered to be the group with the 
# 
# The disparate impact ratios shown in the table above can be used as an example of this rule. A model whose disparate impact ratio is 1 is considered to be perfectly "fair" relative to the sensitive attribute in question. Since neither model in this example has a disparate impact ratio greater than or equal to 1.2 (or <= 0.8), we can say that neither model imposes a significantly disparate impact with respect to gender *according to the Demographic Parity metric*. Although, as will be clear in the next example, "fairness" by one measure does not necessarily mean that a model imposes no disparate impact.
# 

# # Part 4: Testing Other Sensitive Attributes
# 
# ## English Language as a Sensitive Attribute
# Our next experiment will test the presence of bias relative to a patient\'s language. Here we assume that individuals who speak English may be given preferential treatment in an English-speaking society due to the requirement of using a translator. English-speaking ability may also be a proxy for race or religion. As above, we will generate a boolean 'LANGUAGE_ENGL' feature to the baseline data.
# <a id="engldist"></a>

# In[155]:


# Update Split Data to Include Language as a Feature
lang_cols = [c for c in df.columns if c.startswith("LANGUAGE_")]
eng_cols = ['LANGUAGE_ENGL']
X_lang =  df.loc[:,lang_cols]
english_speaking = X_lang[eng_cols].eq(1).any(axis=1)
X_lang.loc[english_speaking, 'LANG_ENGL'] = 1
X_lang.loc[~english_speaking, 'LANG_ENGL'] = 0
X_lang = X_lang.drop(lang_cols, axis=1).fillna(0)
# Show LOS Statistics Relative to LANG_ENGL

display(X_lang.join(df['length_of_stay']).groupby('LANG_ENGL')['length_of_stay'].describe().round(4))
ax_e = df.groupby(english_speaking)['length_of_stay'].plot(kind='kde', title="Probability Density of Length of Stay by English-Speaking Ability")
plt.show()


ttest = stats.ttest_ind(df.loc[english_speaking,'length_of_stay'],df.loc[~english_speaking,'length_of_stay'])
print("\n", "- Results of Independent T-Test -", "\n",
      "T Value:", ttest[0], "\n", "P Value:", ttest[1]
     )


# In[53]:


# Train New Model with Language Feature
X_lang_train = X_train.join(X_lang, how='inner')
X_lang_test = X_test.join(X_lang, how='inner')
lang_model = XGBClassifier(**xgb_params)
lang_model.fit(X_lang_train, y_train.iloc[:,0])
y_pred_lang = lang_model.predict(X_lang_test)
y_prob_lang = lang_model.predict_proba(X_lang_test)


# In[67]:


# Generate Dataframe of  Measure Values for Language-Inclusive Model, Relative to Patient Language
lang_scores = tutorial_helpers.get_aif360_measures_df(X_lang_test, y_test, y_pred_lang, y_prob_lang, sensitive_attributes=['LANG_ENGL'])

# Generate Dataframe of Measure Values for Baseline Model, Relative to Patient Language
lang_ko_scores = tutorial_helpers.get_aif360_measures_df(X_lang_test, y_test, baseline_y_pred, baseline_y_prob, sensitive_attributes=['LANG_ENGL']) 

# Merge Results to Display a Full Comparison
lang_comparison = lang_scores.rename(columns={'Value':'Language Fairness'}
                            ).merge(lang_ko_scores.rename(columns={'Value':'Language Fairness, Baseline (attribute absent)'})
                            )
lang_comparison


# ## Evaluating Justification
# The table above presents the measure values for the two LANG_ENGL-respective models. Although both models appear to be relatively fair by most group-fairness measures, by applying the four-fifths rule it is clear that the Disparate Impact Ratio of the LANG_ENGL-inclusive model shows evidence of disparate impact. Notice that this discrepancy is not clear from the Statistical Parity Difference, which is also a measure for the Disparate Impact metric. In this case the model "favors" the unprivileged class (those who do not speak English), which translates to a longer predicted LOS for those who do not speak English. Even the baseline model, although its Disparate Impact Ratio is within the four-fiths guideline, still shows evidence of some "favortism" toward the non-English speaking group. Does this constitute discrimination? As we noted [above](#disparateimpact), a discriminative disparate impact only exists where differential treatment is unfavorable and unjustified.
# 
# As shown in the distribution figures [above](#engldist), there is a slight difference in the mean values of the English-speaking and non-English speaking groups. Because of this, we may be inclined to believe that the ICU in question could have difficulty in assessing non-English speaking patients due to important information being lost in translation. However, the independent t-test ([above](#engldist)) shows that this difference in means is not statistically significant, so we should not expect a significant difference in the probability of the positive class for either the English-speaking or the non-English speaking group. Hence, the bias induced by the inclusion of LANG_ENGL in our model is not reflective of a real phenomenon. In this case, because there is both evidence of disparate impact and a lack of statistical justification for including the feature, inclusion of LANG_ENGL in our model could be considered discriminatory. However, a lack of statistical justification is not necessary to conclude that the LANG_ENGL model is discriminatory. There could be a historical precedent of non-English speaking bias in the data that has since been addressed; or, LANG_ENGL may be esoterically related to certain diseases, such that it's inclusion makes the model more predictive through its interactions with other features. Neither of these rationale are still likely to be strong enough to justify the inclusion of LANG_ENGL without supporting medical or epidemiological evidence.
# 

# # Part 5: Comparison to FairLearn <a class="anchor" id="part5"></a>
# 
# The FairLearn and AIF360 APIs for Scikit and XGBOOST models are very similar in user experience, and contain a similar set of measures as shown in the table below. Although the set of measures provided by AIF360 is more comprehensive, FairLearn does provide some measures that are unique. First we'll look at FairLearn measures that are also found in AIF360 before explaining the measures that are distinct. 

# In[4]:


# Print Summary Table of Available Measures
print("Summary Table of Available Measures")
tutorial_helpers.print_table_of_measure_availability()


# In[25]:


# Display Example Results for Measures that are Found in AIF360
print("Selection rate", 
      selection_rate(y_test, y_pred_lang) )
print("Demographic parity difference", 
      demographic_parity_difference(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))
print("Demographic parity ratio", 
      demographic_parity_ratio(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL'])) 


# ### Other Measures of Equal Odds
# Two measures that are not present in AIF360 are the Equalized Odds Difference and the Equalized Odds Ratio. Similar to the Average Odds Difference and Average Odds Ratio found in AIF360, these are measures of disparate rates in prediction.
# 
# The Equalized Odds Difference is the greater between the difference in TPR and the difference in FPR. A value of 0 indicates that all groups have the same TPR, FPR, TNR, and FNR, and that the model is "fair" relative to the sensitive attribute.
# > $ equalized\_odds\_difference = max( (FPR_{unprivileged} - FPR_{privileged}), (TPR_{unprivileged} - TPR_{privileged}) )$
# 
# 
# The Equalized Odds Ratio is the smaller between the TPR Ratio and FPR Ratio, where the ratios are defined as the ratio of the smaller of the between-group rates vs the larger of the between-group rates.  A value of 1 means that all groups have the same TPR, FPR, TNR, and FNR.
# > $ equalized\_odds\_ratio = min( \dfrac{FPR_{smaller}}{FPR_{larger}}, \dfrac{TPR_{smaller}}{TPR_{larger}} )$

# In[23]:


print("Equalized Odds Difference",
        equalized_odds_difference(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))
print("Equalized Odds Ratio",
        equalized_odds_ratio(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL']))


# ### Group Summaries
# Similar to AIF360's "difference", which we used [above](#aif_difference_func) to calculate differences in prediction scores, FairLearn provides a "group_summary" function that returns a dictionary with both the overall and the between-group scores for a predictive metric. Also available are convenient wrappers, such as the "balanced_accuracy_score_group_summary" shown below which returns a summary of the balanced accuracy scores.
# 
# To extend the summary functionality, FairLearn also offers a "difference_from_summary" function (shown below), which calculates the between-group prediction difference (again, as we calculated [above](#aif_difference_func)).

# In[24]:


balanced_accuracy_summary = balanced_accuracy_score_group_summary(y_test, y_pred_lang, sensitive_features=X_lang_test['LANG_ENGL'])
print("Balanced Accuracy Summary", balanced_accuracy_summary )
print("Between-Group Balanced Accuracy Difference", difference_from_summary(balanced_accuracy_summary) )


# # Summary
# This tutorial introduced multiple measures of ML fairness in the context of a healthcare model using Scikit-compatible APIs from the AIF360 and FairLearn Python libraries. It discussed the four-fifths rule for evaluating the significance of fairness measures. It demonstrated the value of comparing multiple measures against each other both for a single model and across multiple models. And it discussed reasonable justification for the inclusion of sensitive attributes within a machine learning model.
# 

# # References 
# <a id="barocas2016_ref"></a>
# Barocas S, & Selbst AD (2016). Big data's disparate impact. California Law Review, 104, 671. Available at [http://www.californialawreview.org/wp-content/uploads/2016/06/2Barocas-Selbst.pdf](http://www.californialawreview.org/wp-content/uploads/2016/06/2Barocas-Selbst.pdf) 
# 
# Bellamy RK, Dey K, Hind M, Hoffman SC, Houde S, Kannan K, ... & Nagar S (2018). AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. arXiv Preprint. [arXiv:1810.01943.](https://arxiv.org/abs/1810.01943). See Also [AIF360 Documentation](http://aif360.mybluemix.net/) 
# 
# Bird S, Dudík M,  Wallach H,  & Walker K (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft Research. Available at [https://www.microsoft.com/en-us/research/uploads/prod/2020/05/Fairlearn_whitepaper.pdf](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/Fairlearn_whitepaper.pdf). See Also [FairLearn Reference](https://fairlearn.github.io/).
# 
# <a id="dwork2012_ref"></a>
# Dwork C, Hardt M, Pitassi T, Reingold O, & Zemel R (2012, January). Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference (pp. 214-226). Available at [https://arxiv.org/pdf/1104.3913.pdf](https://arxiv.org/pdf/1104.3913.pdf)
# 
# <a id="fourfifths_ref"></a>
# Equal Employment Opportunity Commission, & Civil Service Commission, Department of Labor & Department of Justice (1978). Uniform guidelines on employee selection procedures. Federal Register, 43(166), 38290-38315. Available at [http://uniformguidelines.com/uniformguidelines.html#18](http://uniformguidelines.com/uniformguidelines.html#18)
# 
# <a id="hardt2016_ref"></a>
# Hardt M, Price E, & Srebro N (2016). Equality of opportunity in supervised learning. In Advances in neural information processing systems (pp. 3315-3323). Available at [http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf](http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf)
# 
# <a id="hcup_ref"></a> 
# Healthcare Cost and Utilization Project (HCUP) (2017, March). HCUP CCS. Agency for Healthcare Research and Quality, Rockville, MD. Available at [www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)
# 
# Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, & Mark RG (2016). Scientific Data. MIMIC-III, a freely accessible critical care database. DOI: 10.1038/sdata.2016.35. Available at [http://www.nature.com/articles/sdata201635](http://www.nature.com/articles/sdata201635)
# 
# <a id="kim2018_ref"></a>   
# Kim M, Reingol O, & Rothblum G (2018). Fairness through computationally-bounded awareness. In Advances in Neural Information Processing Systems (pp. 4842-4852).
# Available at [https://arxiv.org/pdf/1803.03239.pdf](https://arxiv.org/pdf/1803.03239.pdf)  
#   
# <a id="romei2014_ref"></a>  
# Romei, A., & Ruggieri, S. (2014). A multidisciplinary survey on discrimination analysis. The Knowledge Engineering Review, 29(5), 582-638. Available at [https://www.cambridge.org/core/journals/knowledge-engineering-review/article/multidisciplinary-survey-on-discrimination-analysis/D69E925AC96CDEC643C18A07F2A326D7](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/multidisciplinary-survey-on-discrimination-analysis/D69E925AC96CDEC643C18A07F2A326D7)
#   
# <a id="russell2017_ref"></a>
# Russell C, Kusner MJ, Loftus J, & Silva R (2017). When worlds collide: integrating different counterfactual assumptions in fairness. In Advances in Neural Information Processing Systems (pp. 6414-6423). Available at [https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf](https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf)  
# 
# <a id="shorrocks_ref"></a>
# Shorrocks AF (1980). The class of additively decomposable inequality measures. Econometrica: Journal of the Econome)tric Society, 613-625. Available at [http://www.vcharite.univ-mrs.fr/PP/lubrano/atelier/shorrocks1980.pdf](http://www.vcharite.univ-mrs.fr/PP/lubrano/atelier/shorrocks1980.pdf
# 
# <a id="speicher2018_ref"></a>
# Speicher T, Heidari H, Grgic-Hlaca N, Gummadi KP, Singla A, Weller A, & Zafar M B (2018, July). A unified approach to quantifying algorithmic unfairness: Measuring individual &group unfairness via inequality indices. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2239-2248). Available at [https://arxiv.org/pdf/1807.00787.pdf](https://arxiv.org/pdf/1807.00787.pdf)
# 
# <a id="zemel2013_ref"></a>
# Zemel R, Wu Y, Swersky K, Pitassi T, & Dwork C (2013, February). Learning fair representations. In International Conference on Machine Learning (pp. 325-333).
# 
# <a id="zafar2017_ref"></a>
# Zafar MB, Valera I, Gomez Rodriguez, M, & Gummadi KP (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).  https://arxiv.org/pdf/1610.08452.pdf
# 

# # Additional Resources and Tutorials
# 
# [Fairness Indicators](https://www.tensorflow.org/tfx/fairness_indicators) - A TensorFlow-enabled suite designed for simulation of long-term effects caused by the implementation of a machine learning model.   
# 
# ["A Tutorial on Fairness in Machine Learning" by Ziyuan Zhong](https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb) - A popular reference with additional background on fairness metrics.
#    
# ["How to define fairness to detect and prevent discriminatory outcomes in Machine Learning" by Valeria Cortez](https://towardsdatascience.com/how-to-define-fairness-to-detect-and-prevent-discriminatory-outcomes-in-machine-learning-ef23fd408ef2#:~:text=Demographic%20Parity%20states%20that%20the,%E2%80%9Cbeing%20shown%20the%20ad%E2%80%9D) - Another source for background on fairness metrics.
#    
#   
