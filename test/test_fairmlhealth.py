'''
Validation tests forfairMLHealth
'''
from fairmlhealth import model_comparison as fhmc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


def synth_dataset():
    df = pd.DataFrame({'A': [1, 2, 50, 3, 45, 32],
                       'B': [34, 26, 44, 2, 1, 1],
                       'C': [32, 23, 34, 22, 65, 27],
                       'gender': [0, 1, 0, 1, 1, 0],
                       'target': [1, 0, 0, 1, 0, 1]
                       })
    return df


def test_compare_func():
    # Synthesize data
    df = synth_dataset()
    X = df[['A', 'B', 'C', 'gender']]
    y = df[['target']]
    splits = train_test_split(X, y, test_size=0.75, random_state=36)
    X_train, X_test, y_train, y_test = splits

    # Train models
    model_1 = BernoulliNB().fit(X_train, y_train)
    model_2 = DecisionTreeClassifier().fit(X_train, y_train)

    # Generate test arguments
    prtc_attr = X_test['gender']
    model_dict = {'model_1': model_1, 'model_2': model_2}

    # Generate comparison
    test1 = fhmc.compare_measures(X_test, y_test, prtc_attr, model_dict)
    assert test1 is not None
    test2 = fhmc.compare_measures(X_test, y_test, prtc_attr, [model_1])
    assert test2 is not None
    test3 = fhmc.compare_measures(X_test, y_test, prtc_attr, None)
#
