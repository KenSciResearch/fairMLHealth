'''
A quick-running script to identify syntax issues and basic errors in model
measurement during development (used to catch problems that are missed by the
linter)
'''


from fairmlhealth import model_comparison as fhmc
from fairmlhealth.reports import flag
import logging
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


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
model_2 = DecisionTreeClassifier().fit(X_train, y_train)

# Deterimine your set of protected attributes
prtc_attr = X_test['gender']

# Test that measure_model
fhmc.measure_model(X_test, y_test, prtc_attr, model_1)


# Pass the data and models to the compare models function, as above
fhmc.compare_models(X_test, y_test, prtc_attr,
                                  {'model_1': model_1, 'model_2': model_2})


# Report success
logging.info("Passed Basic Model Comparison Test")
