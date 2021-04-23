'''
'''


from fairmlhealth import reports
import numpy as np
import pytest
import pandas as pd


 # ToDo: Add more robust testing throughout

@pytest.fixture(scope="class")
def load_classification_data(request):
    # Arrays that must be specific for testing
    idx = list(range(0, 8))
    pa = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="prtc_attr")
    targ_data = [0, 0, 1, 1, 0, 0, 1, 1]
    y = pd.Series(targ_data, index=idx, name="y")
    prfct_fair = pd.Series(targ_data, index=idx, name="prfct")
    avg_fair = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=idx, name="avg")
    #
    np.random.seed(547)
    df = pd.DataFrame({'A': np.random.randint(1, 4, 8),
                       'B': np.random.randint(1, 8, 8),
                       'C': np.random.randint(1, 16, 8),
                       'D': np.random.uniform(-10, 10, 8)
                       })
    df = pd.concat([df, pa, y, prfct_fair, avg_fair], axis=1)

    request.cls.df = df
    yield


@pytest.fixture(scope="class")
def load_regression_data(request):
    # Arrays that must be specific for testing
    idx = list(range(0, 8))
    pa = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="prtc_attr")
    targ_data = np.random.uniform(-10, 10, 8)
    y = pd.Series(targ_data, index=idx, name="y")
    prfct_fair = pd.Series(targ_data, index=idx, name="prfct")
    avg_fair = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=idx, name="avg")
    avg_fair[avg_fair.eq(1)] = targ_data
    #
    np.random.seed(547)
    df = pd.DataFrame({'A': np.random.randint(1, 2, 8),
                       'B': np.random.randint(1, 4, 8),
                       'C': np.random.randint(1, 16, 8),
                       'D': np.random.uniform(-2, 2, 8)
                       })
    df = pd.concat([df, pa, y, prfct_fair, avg_fair], axis=1)

    request.cls.df = df
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

@pytest.mark.usefixtures("load_classification_data")
class TestDataReport:
    """ Developmental class to validate that non-standard inputs are processed
        as expected. Will be finalized for release of V2.0
    """
    # ToDo: Add more robust testing
    def __dev_test_without_y(self):
        with pytest.raises(Exception):
                _ = reports.data_report(self.df)

