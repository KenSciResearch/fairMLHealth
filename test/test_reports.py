'''
'''
 # ToDo: Add more robust testing throughout


from fairmlhealth import reports
import numpy as np
import pytest
import pandas as pd
from .__utils import synth_dataset
np.random.seed(547)


@pytest.fixture(scope="class")
def load_classification_data(synth_dataset, request):
    df = synth_dataset
    targ_data = [0, 0, 1, 1, 0, 0, 1, 1]
    y = pd.Series(targ_data, index=df.index, name="y")
    prfct_fair = pd.Series(targ_data, index=df.index, name="prfct")
    avg_fair = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=df.index, name="avg")
    #
    output = pd.concat([df, y, prfct_fair, avg_fair], axis=1)
    request.cls.df = output
    yield


@pytest.fixture(scope="class")
def load_regression_data(synth_dataset, request):
    df = synth_dataset
    targ_data = np.random.uniform(-10, 10, 8)
    y = pd.Series(targ_data, index=df.index, name="y")
    prfct_fair = pd.Series(targ_data, index=df.index, name="prfct")
    avg_fair = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=df.index, name="avg")
    avg_fair[avg_fair.eq(1)] = targ_data
    #
    output = pd.concat([df, y, prfct_fair, avg_fair], axis=1)
    request.cls.df = output
    yield


@pytest.mark.usefixtures("load_classification_data")
class TestStandardCassificationReports:
    """ Validates that standard inputs are processed without error
    """
    def test_classification_summary(self):
        _ = reports.summary_report(self.df, self.df['prtc_attr'], self.df['y'],
                                   self.df['avg'], pred_type="classification")

    def test_classification_data_report(self):
        _ = reports.data_report(self.df, self.df['y'])

    def test_classification_performance_report(self):
        _ = reports.performance_report(self.df, self.df['y'], self.df['avg'],
                                       pred_type="classification")

    def test_classification_bias_report(self):
        _ = reports.bias_report(self.df, self.df['y'], self.df['avg'],
                                pred_type="classification")


@pytest.mark.usefixtures("load_regression_data")
class TestStandardRegressionReports:
    """ Developmental class to validate that standard inputs are processed
        without error. Will be finalized for release of V2.0
    """
    def __dev_test_regression_summary(self):
        _ = reports.summary_report(self.df, self.df['prtc_attr'], self.df['y'],
                                   self.df['avg'], pred_type="regression")

    def __dev_test_regression_data_report(self):
        _ = reports.data_report(self.df, self.df['y'])

    def __dev_test_regression_performance_report(self):
        _ = reports.performance_report(self.df, self.df['y'], self.df['avg'],
                                  pred_type="regression")

    def __dev_test_regression_bias_report(self):
        _ = reports.bias_report(self.df, self.df['y'], self.df['avg'],
                            pred_type="regression")


@pytest.mark.usefixtures("load_regression_data")
class TestDataReport:
    """ Developmental class to validate that non-standard inputs are processed
        as expected. Will be finalized for release of V2.0
    """
    # ToDo: Add more robust testing
    def test_missing_y(self):
        with pytest.raises(Exception):
            _ = reports.data_report(self.df, None)

    def test_y_as_df(self):
        _ = reports.data_report(self.df, self.df,
                                features=['A', 'B', 'C'], targets=['C','D'])
