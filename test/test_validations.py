'''
Validation tests for fairmlhealth
'''
from fairmlhealth import model_comparison as fhmc
from fairmlhealth.utils import ValidationError

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

import pytest
import pandas as pd


@pytest.fixture(scope="module")
def synth_dataset():
    df = pd.DataFrame({'A': [1, 2, 50, 3, 45, 32, 23],
                       'B': [34, 26, 44, 2, 1, 1, 12],
                       'C': [32, 23, 34, 22, 65, 27, 11],
                       'gender': [0, 1, 0, 1, 1, 0, 0],
                       'age': [0, 1, 1, 1, 1, 0, 1],
                       'target': [1, 0, 0, 1, 0, 1, 0]
                       })
    return df


@pytest.fixture(scope="class")
def load_data(synth_dataset, request):
    df = synth_dataset
    X = df[['A', 'B', 'C', 'gender', 'age']]
    y = df[['target']]
    splits = train_test_split(X, y, test_size=0.75, random_state=36)
    X_train, X_test, y_train, y_test = splits

    # Train models
    model_1 = BernoulliNB().fit(X_train, y_train)
    model_2 = DecisionTreeClassifier().fit(X_train, y_train)

    # Set test attributes
    request.cls.X = X_test
    request.cls.y = y_test
    request.cls.prtc_attr = X_test['gender']
    request.cls.model_dict = {0: model_1, 1: model_2}
    yield


@pytest.mark.usefixtures("load_data")
class TestCMFunction:
    """ Test proper functioning of the compare_measures function
    """
    def test_compare_valid_inputs(self):
        result = fhmc.compare_measures(self.X, self.y, self.prtc_attr,
                                       self.model_dict)
        assert result is not None

    def test_compare_with_model_as_array(self):
        result = fhmc.compare_measures(self.X, self.y, self.prtc_attr,
                                       [self.model_dict[0]])
        assert result is not None

    def test_compare_with_model_as_none(self):
        result = fhmc.compare_measures(self.X, self.y, self.prtc_attr, None)
        assert result is not None

    def test_compare_with_protected_attributes(self):
        result = fhmc.compare_measures([self.X, self.X], [self.y, self.y],
                                       [self.prtc_attr, self.prtc_attr],
                                       [self.model_dict[0], self.model_dict[1]])
        assert result is not None


@pytest.mark.usefixtures("load_data")
class TestCMValidations:
    """ Validations for the compare_measures function
    """
    def test_incorrect_length_inputs(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {1: self.prtc_attr},
                                  self.model_dict)

    def test_incompatible_iterables(self):
        with pytest.raises(ValidationError):
            with pytest.raises(Exception):
                fhmc.compare_measures([self.X, self.X], [self.y, self.y],
                                      [self.prtc_attr, self.prtc_attr],
                                       self.model_dict)
            with pytest.raises(Exception):
                fhmc.compare_measures([self.X, self.X], {0: self.y, 1: self.y},
                                      {0: self.prtc_attr, 1: self.prtc_attr},
                                      self.model_dict)

    def test_invalid_X_member(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  {0: self.y, 1: self.y})

    def test_invalid_y_member(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: None},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

    def test_invalid_prtc_member(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: None},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

    def test_invalid_model_member(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: None},
                                  self.model_dict)

    def test_differing_X_keys(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({5: self.X, 6: self.X},
                                  {0: self.y, 1: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

    def test_differing_y_keys(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {5: self.y, 6: self.y},
                                  {0: self.prtc_attr, 1: self.prtc_attr},
                                  self.model_dict)

    def test_differing_prtc_keys(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  {5: self.prtc_attr, 6: self.prtc_attr},
                                  self.model_dict)

    def test_missing_prtc_keys(self):
        with pytest.raises(Exception):
            fhmc.compare_measures({0: self.X, 1: self.X},
                                  {0: self.y, 1: self.y},
                                  None,
                                  self.model_dict)

