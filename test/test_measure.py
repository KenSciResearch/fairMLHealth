"""
"""


from fairmlhealth import measure
import numpy as np
import pytest
import pandas as pd
from .__utils import synth_dataset

np.random.seed(547)


@pytest.fixture(scope="class")
def load_classification_data(request):
    df = synth_dataset(32)
    X = df[["A", "B", "C", "D", "prtc_attr", "other"]]
    y = df["binary_target"].rename("y")
    avg_fair = df["avg_binary_pred"].rename("avg")
    #
    data = pd.concat([X, y, avg_fair], axis=1)
    request.cls.df = data
    yield


@pytest.fixture(scope="class")
def load_regression_data(request):
    df = synth_dataset(32)
    X = df[["A", "B", "C", "D", "E", "prtc_attr", "other"]]
    y = df["continuous_target"].rename("y")
    avg_fair = df["avg_cont_pred"].rename("avg")
    #
    data = pd.concat([X, y, avg_fair], axis=1)
    request.cls.df = data
    yield


@pytest.mark.usefixtures("load_classification_data")
class TestStandardCassificationReports:
    """ Validates that standard inputs are processed without error
    """

    def test_classification_summary(self):
        _ = measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["y"],
            self.df["avg"],
            pred_type="classification",
        )

    def test_classification_data(self):
        _ = measure.data(self.df, self.df["y"])

    def test_classification_performance(self):
        _ = measure.performance(
            self.df, self.df["y"], self.df["avg"], pred_type="classification"
        )

    def test_classification_bias(self):
        _ = measure.bias(
            self.df, self.df["y"], self.df["avg"], pred_type="classification"
        )


@pytest.mark.usefixtures("load_regression_data")
class TestStandardRegressionReports:
    """ Developmental class to validate that standard inputs are processed
        without error. Will be finalized for release of V2.0
    """

    def __dev_test_regression_summary(self):
        _ = measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["y"],
            self.df["avg"],
            pred_type="regression",
        )

    def __dev_test_regression_data(self):
        _ = measure.data(self.df, self.df["y"])

    def __dev_test_regression_performance(self):
        _ = measure.performance(
            self.df, self.df["y"], self.df["avg"], pred_type="regression"
        )

    def __dev_test_regression_bias(self):
        _ = measure.bias(self.df, self.df["y"], self.df["avg"], pred_type="regression")


@pytest.mark.usefixtures("load_regression_data")
class TestDataReport:
    """ Developmental class to validate that non-standard inputs are processed
        as expected. Will be finalized for release of V2.0
    """

    # ToDo: Add more robust testing
    def test_missing_y(self):
        with pytest.raises(Exception):
            _ = measure.data(self.df, None)

    def test_y_as_df(self):
        _ = measure.data(self.df, self.df, features=["A", "B", "C"], targets=["C", "D"])
