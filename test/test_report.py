'''
'''


from fairmlhealth import report
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import pytest
import pandas as pd
from .__testing_utilities import synth_dataset



@pytest.fixture(scope="class")
def load_data(request):
    N= 128
    df = synth_dataset(N)
    X = df[['A', 'B', 'C', 'E', 'F', 'prtc_attr', 'other']]

    y = pd.Series((X['F']*X['other']).values + np.random.randint(0, 2, N), name='y').clip(upper=1)
    X_train, y_train= X.iloc[0:int(N/4)], y.iloc[0:int(N/4)]
    X_test, y_test = X.iloc[int(N/4):N], y.iloc[int(N/4):N]

    # Train models
    model_1 = BernoulliNB().fit(X_train, y_train)
    model_2 = DecisionTreeClassifier().fit(X_train, y_train)
    model_3 = DecisionTreeClassifier().fit(X_train.to_numpy(),
                                           y_train.to_numpy())

    # Set test attributes
    request.cls.X = X_test
    request.cls.y = y_test
    request.cls.prtc_attr = X_test['prtc_attr']
    request.cls.model_dict = {0: model_1, 1: model_2, 2: model_3}
    yield


@pytest.mark.usefixtures("load_data")
class TestCompModFunction:
    """ Test proper functioning of the compare function. Result
        should be a pandas dataframe
    """
    def is_result_valid(self, result):
        if not isinstance(result, pd.DataFrame) and result.shape[0] > 0:
            raise AssertionError("Invalid Result")

    def test_single_dataInputs(self):
        result = report.compare(self.X, self.y, self.prtc_attr,
                                     self.model_dict, flag_oor=False)
        self.is_result_valid(result)

    def test_model_list(self):
        result = report.compare(self.X, self.y, self.prtc_attr,
                                     [self.model_dict[0]], flag_oor=False)
        self.is_result_valid(result)

    def test_mixed_groupings(self):
        result = report.compare([self.X, self.X],
                                     self.y, self.prtc_attr,
                                     [self.model_dict[0], self.model_dict[1]],
                                     flag_oor=False)
        self.is_result_valid(result)

    def test_with_protected_attributes(self):
        result = report.compare([self.X, self.X], [self.y, self.y],
                                     [self.prtc_attr, self.prtc_attr],
                                     [self.model_dict[0], self.model_dict[1]],
                                     flag_oor=False)
        self.is_result_valid(result)

    def test_preds_not_models(self):
        result = report.compare([self.X, self.X],
                                     self.y, self.prtc_attr,
                                     predictions=[self.y, self.y],
                                     flag_oor=False)
        self.is_result_valid(result)

    def test_preds_and_probs(self):
        result = report.compare([self.X, self.X],
                                     self.y, self.prtc_attr,
                                     predictions=[self.y, self.y],
                                     probabilities=[self.y, self.y],
                                     flag_oor=False)
        self.is_result_valid(result)

    def test_multiple_calls(self):
        args = (self.X, self.y, self.prtc_attr, self.model_dict[0])
        _ = report.compare(*args, flag_oor=False)
        result = report.compare(*args, flag_oor=False)
        self.is_result_valid(result)


@pytest.mark.usefixtures("load_data")
class TestCompModValidations:
    """ Validations for the compare function
    """
    def test_mismatch_input_numbers(self):
        with pytest.raises(Exception):
            report.compare({0: self.X, 1: self.X},
                                {0: self.y, 1: self.y},
                                {1: self.prtc_attr},
                                self.model_dict,
                                flag_oor=False)

    def test_missing_models(self):
        with pytest.raises(Exception):
            report.compare({0: self.X, 1: self.X},
                                {0: self.y, 1: self.y},
                                {0: self.prtc_attr, 1: self.prtc_attr},
                                {0: None, 1: None},
                                flag_oor=False)

    def test_invalid_X_member(self):
        with pytest.raises(Exception):
            report.compare({0: self.X, 1: self.X},
                                {0: self.y, 1: self.y},
                                {0: self.prtc_attr, 1: self.prtc_attr},
                                {0: self.y, 1: self.y},
                                flag_oor=False)

    def test_invalid_y_member(self):
        with pytest.raises(Exception):
            report.compare({0: self.X, 1: None},
                                {0: self.y, 1: self.y},
                                {0: self.prtc_attr, 1: self.prtc_attr},
                                self.model_dict,
                                flag_oor=False)

    def test_invalid_prtc_member(self):
        with pytest.raises(Exception):
            report.compare({0: self.X, 1: self.X},
                                {0: self.y, 1: None},
                                {0: self.prtc_attr, 1: self.prtc_attr},
                                self.model_dict,
                                flag_oor=False)

    def test_invalid_model_member(self):
        with pytest.raises(Exception):
            report.compare({0: self.X, 1: self.X},
                                {0: self.y, 1: self.y},
                                {0: self.prtc_attr, 1: None},
                                self.model_dict,
                                flag_oor=False)

    def test_differing_keys(self):
        with pytest.raises(Exception):
            report.compare({5: self.X, 6: self.X},
                                {0: self.y, 1: self.y},
                                {0: self.prtc_attr, 1: self.prtc_attr},
                                self.model_dict,
                                flag_oor=False)

    def test_missing_keys(self):
        with pytest.raises(Exception):
            report.compare({0: self.X, 1: self.X},
                                {0: self.y, 1: self.y},
                                None,
                                self.model_dict,
                                flag_oor=False)
