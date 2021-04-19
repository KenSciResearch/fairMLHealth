'''
This is a quick-running script to identify syntax issues and basic errors in
the model_comparison during development. It can be used to catch problems that
are missed by the linter before running the full set of unit tests.
'''


from fairmlhealth import model_comparison as fhmc, stratified_reports
import logging
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


# Load data
X = pd.DataFrame({'col1':[1,2,50,3,45,32,44,35],
                  'col2':[34,26,44,2,1,1,2,4],
                  'col3':[32,23,34,22,65,27,44,27],
                  'gender':[0,1,0,1,1,0,1,0],
                  'ethnicity':[0,0,0,1,1,1,1,1]
                 })
y = pd.DataFrame({'y':[1,0,0,1,0,1,1,1]})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75,
                                                    random_state=36)

#Train models
model_1 = BernoulliNB().fit(X_train, y_train)

# Deterimine your set of protected attributes
prtc_attr = X_test['gender']


# Arguments Option 1: pass full set of data, subsetting with *features* argument
stratified_reports.data_report(X_test, y_test, features=['gender'])

# Arguments Option 2: pass the data subset of interest without using the *features* argument
stratified_reports.data_report(X_test[['gender']], y_test)

stratified_reports.classification_performance(X_test[['gender']], y_test, model_1.predict(X_test))

stratified_reports.classification_fairness(X_test[['gender', 'col1']], y_test, model_1.predict(X_test))

# Report success
logging.info("Passed Basic Model Comparison Test")
