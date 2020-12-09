'''
Validation tests forfairMLHealth
'''
from fairmlhealth import model_comparison as fhmc
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import unittest
from fairmlhealth.utils import ValidationError


def synth_dataset():
    df = pd.DataFrame({'A': [1, 2, 50, 3, 45, 32, 23],
                       'B': [34, 26, 44, 2, 1, 1, 12],
                       'C': [32, 23, 34, 22, 65, 27, 11],
                       'gender': [0, 1, 0, 1, 1, 0, 0],
                       'age': [0, 1, 1, 1, 1, 0, 1],
                       'target': [1, 0, 0, 1, 0, 1, 0]
                       })
    return df


class TestCompareFunc(unittest.TestCase):
    is_loaded = False

    def load_data(self):
        if self.is_loaded:
            pass
        else:
            # Synthesize data
            df = synth_dataset()
            X = df[['A', 'B', 'C', 'gender', 'age']]
            y = df[['target']]
            splits = train_test_split(X, y, test_size=0.75, random_state=36)
            X_train, X_test, y_train, y_test = splits

            # Train models
            model_1 = BernoulliNB().fit(X_train, y_train)
            model_2 = DecisionTreeClassifier().fit(X_train, y_train)

            # Set test attributes
            self.X = X_test
            self.y = y_test
            self.prtc_attr = X_test['gender']
            self.model_dict = {'model_1': model_1, 'model_2': model_2}
            self.is_loaded = True

    def run_all(self):
        self.load_data()
        self.test_compare_func()
        self.test_validation()

    def test_compare_func(self):
        ''' Validates the compare_measures function on allowable inputs. All
                sub-tests should return non-NoneType object.
        '''
        self.load_data()
        X, y, prtc_attr = self.X, self.y, self.prtc_attr
        model_1, model_2 = self.model_dict.values()

        # Generate comparison
        test1 = fhmc.compare_measures(X, y, prtc_attr, self.model_dict)
        test2 = fhmc.compare_measures(X, y, prtc_attr, [model_1])
        test3 = fhmc.compare_measures(X, y, prtc_attr, None)
        test4 = fhmc.compare_measures([X, X], [y, y], [prtc_attr, prtc_attr],
                                      [model_1, model_2])

        assert not any(t is None for t in [test1, test2, test3, test4])

    def test_validation(self):
        ''' Tests that the FairCompare object validation method. All sub-tests
                should raise errors.
        '''
        self.load_data()
        X, y, prtc_attr = self.X, self.y, self.prtc_attr
        model_dict = self.model_dict
        model_1, model_2 = model_dict.values()

        #
        with self.assertRaises(ValidationError):
            fhmc.compare_measures([X, X], [y], [prtc_attr], [model_1, model_2])
        with self.assertRaises(ValidationError):
            fhmc.compare_measures({0: X, 1: X}, {0: y, 1: y},
                                  {0: prtc_attr}, {0: model_1, 1: model_1})
        with self.assertRaises(ValidationError):
            fhmc.compare_measures({0: X, 1: X}, {0: y, 1: y},
                                  {0: prtc_attr}, {99: y, 50: y})
        with self.assertRaises(ValidationError):
            fhmc.compare_measures([X, X], [y, y],
                                  [prtc_attr, prtc_attr],
                                  {0: model_1, 1: model_1})
        with self.assertRaises(ValidationError):
            fhmc.compare_measures([X, X], {0: y, 1: y},
                                  {0: prtc_attr, 1: prtc_attr},
                                  {0: model_1, 1: model_1})
        with self.assertRaises(ValidationError):
            fhmc.compare_measures([X, X], [y, y],
                                  None, {0: model_1, 1: model_1})
        with self.assertRaises(ValidationError):
            fhmc.compare_measures({0: X, 1: None}, {0: y, 1: y},
                                  {0: prtc_attr, 1: prtc_attr},
                                  {0: model_1, 1: model_1})


if __name__ == '__main__':
    tester = TestCompareFunc()
    tester.run_all()
    print("Passed All Tests")
