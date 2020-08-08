'''

Note: Attributes in the input data are tagged according to the feature index with
    an additional prefix indicating the variable type (see also 'tagged_fature_id'
    in the feature index).
    Prefixes are as follows:
        A - age
        E - ethnicity
        D - diagnosis
        G - gender
        I - insurance
        L - language
        M - marital status
        P - procedure
        R - religion
AIF360 Metrics:
    # meta-metrics
    'difference', 'ratio',
    # scorer factory
    'make_scorer', 
    # helpers
    'specificity_score', 'base_rate', 'selection_rate', 'generalized_fpr',
    'generalized_fnr',
    # group fairness
    'statistical_parity_difference', 'disparate_impact_ratio',
    'equal_opportunity_difference', 'average_odds_difference',
    'average_odds_error',
    # individual fairness
    'generalized_entropy_index', 'generalized_entropy_error',
    'between_group_generalized_entropy_error', 'theil_index',
    'coefficient_of_variation', 'consistency_score',
    # aliases
    'sensitivity_score', 'mean_difference', 'false_negative_rate_error',
    'false_positive_rate_error'

'''

import os
import pandas as pd
import sys

# Prediction Libs
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from xgboost import XGBClassifier, XGBRegressor, plot_tree

sys.path.append(os.path.expanduser("~/repos/Scratch/camagallen"))
from prediction_tester import predictions as preds

from aif360.sklearn.metrics import disparate_impact_ratio
from aif360.sklearn.metrics import *



def print_scores(y_test, y_pred, y_test_lbl, y_pred_lbl):
    """ Prints AIF360 measurements for the model results
    """
    ## Group Fairness
    print("base_rate",
        base_rate(y_test_lbl, y_pred_lbl)
    )
    print("selection_rate",
        selection_rate(y_test_lbl, y_pred_lbl)
    )
    print("disparate_impact_ratio",
        disparate_impact_ratio(y_test_lbl, y_pred_lbl, prot_attr='is_male')
    )
    print("average_odds_difference",
        average_odds_difference(y_test_lbl, y_pred_lbl, prot_attr='is_male')
    )
    print("average_odds_error",
        average_odds_error(y_test_lbl, y_pred_lbl, prot_attr='is_male')
    )
    print("equal_opportunity_difference",
        equal_opportunity_difference(y_test_lbl, y_pred_lbl, prot_attr='is_male')
    )
    print("statistical_parity_difference",
        statistical_parity_difference(y_test_lbl, y_pred_lbl, prot_attr='is_male')
    ) 
    ## individual fairness
    print("generalized_entropy_index",
        generalized_entropy_index(y_pred)
    )
    print("generalized_entropy_error",
        generalized_entropy_error(y_test, y_pred)
    )
    print("between_group_generalized_entropy_error",
        between_group_generalized_entropy_error(y_test_lbl, y_pred_lbl, prot_attr='is_male')
    )
    print("theil_index",
        theil_index(y_pred)
    )
    print("coefficient_of_variation",
        coefficient_of_variation(y_pred)
    )
    print("consistency_score",
        consistency_score(X_test, y_pred)
    )




#
# target columns = ['los_ks', 'los_value', 'los_bin', 'los_median']
uid_col = 'HADM_ID'
data_file = "matched_input_data.csv"

# Load and split data, keeping only the targets required for the test
df = pd.read_csv(data_file)
targ_df = df[['HADM_ID', 'los_bin', 'los_value']]
df = df.loc[:,[c for c in df.columns if not c.startswith('los_') or c == 'los_bin']]

# Keep only one sensitive attribute ('G_f582', aka. "male/non-male")
drop_cols = [c for c in df.columns if c[0] in ['E', 'R', 'I', 'L']] + ['G_f581']
df = df.loc[:,[c for c in df.columns if c not in drop_cols]]
df.rename(columns={'G_f582':'is_male'}, inplace=True)

# Split data using custom splitter
split_data = preds.split_data(df, target_col='los_bin', split_frac=0.8, uid_col=uid_col)
X_train, X_test, y_train, y_test, test_uids = split_data

# Set model
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_test))
y_test_lbl = pd.concat([X_test[['is_male']], y_test], axis=1)
y_pred_lbl = pd.concat([X_test[['is_male']].reset_index(drop=True), y_pred], axis=1)
y_pred_lbl.columns = ['is_male', 'los_bin']

y_test_lbl.set_index('is_male', inplace=True)
y_pred_lbl.set_index('is_male', inplace=True)


print("\nWith is_male:")
print_scores(y_test, y_pred, y_test_lbl, y_pred_lbl)


# generate alternative model
ko_model = XGBClassifier()
ko_model.fit(X_train.drop('is_male', axis=1), y_train)
ko_y_pred = pd.Series( ko_model.predict(X_test.drop('is_male', axis=1)) )
ko_y_test_lbl = pd.concat([X_test[['is_male']], y_test], axis=1)
ko_y_pred_lbl = pd.concat([X_test[['is_male']].reset_index(drop=True), ko_y_pred], axis=1)
ko_y_pred_lbl.columns = ['is_male', 'los_bin']

ko_y_test_lbl.set_index('is_male', inplace=True)
ko_y_pred_lbl.set_index('is_male', inplace=True)

print("\nWithout is_male:")
print_scores(y_test, ko_y_pred, ko_y_test_lbl, y_pred_lbl)

sys.exit()