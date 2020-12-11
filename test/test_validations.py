'''
Validation tests forfairMLHealth
'''
from fairmlhealth import model_comparison as fhmc
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
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


class TestValidations:
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
            self.model_dict = {0: model_1, 1: model_2}
            self.is_loaded = True

    ''' Validate the compare_measures function on allowable inputs. All
        sub-tests should return non-NoneType object.
    '''
    def test_compare_func_1(self):
        """ All data args as data (not iterable) with models in dict should
            return a dataframe
        """
        self.load_data()
        test1 = fhmc.compare_measures(self.X, self.y, self.prtc_attr,
                                      self.model_dict)
        assert isinstance(test1, pd.DataFrame) and test1.shape[0] > 0

    def test_compare_func_2(self):
        """ All args outside of lists should return a dataframe """
        self.load_data()
        test2 = fhmc.compare_measures(self.X, self.y, self.prtc_attr,
                                      self.model_dict[0])
        assert isinstance(test2, pd.DataFrame) and test2.shape[0] > 0

    def test_compare_func_3(self):
        """ All data args as data (not iterable) but None models should return
            an empty data frame
        """
        self.load_data()
        test3 = fhmc.compare_measures(self.X, self.y, self.prtc_attr, None)
        assert isinstance(test3, pd.DataFrame) and test3.shape[0] == 0

    def test_compare_func_4(self):
        """ All args in equal length lists should return a dataframe """
        self.load_data()
        test4 = fhmc.compare_measures([self.X, self.X], [self.y, self.y],
                                      [self.prtc_attr, self.prtc_attr],
                                      [self.model_dict[0],
                                       self.model_dict[1]])
        assert isinstance(test4, pd.DataFrame) and test4.shape[0] > 0

    ''' Tests that the FairCompare object validation method. All sub-tests
        should raise errors.
    '''
    def test_validation_1(self):
        """ Test for one data arg of long list length """
        self.load_data()
        with pytest.raises(Exception):
            fhmc.compare_measures([self.X, self.X], [self.y], [self.prtc_attr],
                                  [self.model_dict[0], self.model_dict[1]])

    def test_validation_2(self):
        """ Test for one data arg of long length """
        self.load_data()
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {1: self.prtc_attr},
                                  self.model_dict)

    def test_validation_3(self):
        """ Test for incompatible list/dict """
        self.load_data()
        with pytest.raises(Exception):
            fhmc.compare_measures([self.X, self.X], [self.y, self.y],
                                  [self.prtc_attr, self.prtc_attr],
                                  self.model_dict)

        with pytest.raises(Exception):
            fhmc.compare_measures([self.X, self.X], {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

    def test_validation_4(self):
        """ Test for one model dict without non-model data """
        self.load_data()
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  {0: self.y, 1: self.y})

    def test_validation_5(self):
        """ Test for differing dict keys """
        self.load_data()
        with pytest.raises(Exception):
            fhmc.compare_measures({5: self.X, 6: self.X},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {5: self.y, 6: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {5: self.prtc_attr, 6: self.prtc_attr},
                                  self.model_dict)

    def test_validation_6(self):
        """ Test for missing argument """
        self.load_data()
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  None,
                                  self.model_dict)

    def test_validation_7(self):
        """ Test for missing sub-argument """
        self.load_data()
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: None},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: None},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: None},
                                  self.model_dict)
