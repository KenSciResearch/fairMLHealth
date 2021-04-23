'''
'''


from fairmlhealth import reports
import numpy as np
import pytest
import pandas as pd



@pytest.fixture(scope="class")
def load_classification_data(request):
    # Arrays that must be specific for testing
    idx = list(range(0, 8))
    pa = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="protected_attribute")
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
    pa = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="protected_attribute")
    targ_data = np.random.uniform(-10, 10, 8)
    y = pd.Series(targ_data, index=idx, name="y")
    prfct_fair = pd.Series(targ_data, index=idx, name="prfct")
    avg_fair = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=idx, name="avg")
    avg_fair[avg_fair.eq(1)] = targ_data
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


@pytest.mark.usefixtures("load_classification_data")
class TestClassificationReports:
    """
    """
    # ToDo: Add more robust testing
    def test_classification_summary(self):
        result = reports.summary_report(self.df, self.df['y'],
                                        self.df['avg_fair'],
                                        type="classification")

    def test_classification_data_report(self):
        result = reports.data_report(self.df, self.df['y'])

    def test_classification_performance_report(self):
        result = reports.performance_report(self.df, self.df['y'],
                                        self.df['avg_fair'],
                                        type="classification")

    def test_classification_bias_report(self):
        result = reports.bias_report(self.df, self.df['y'],
                                    self.df['avg_fair'],
                                    type="classification")


@pytest.mark.usefixtures("load_regression_data")
class TestRegressionReports:
    """
    """
    # ToDo: Add more robust testing
    def test_regression_summary(self):
        result = reports.summary_report(self.df, self.df['y'],
                                        self.df['avg_fair'],
                                        type="regression")

    def test_regression_data_report(self):
        result = reports.data_report(self.df, self.df['y'])

    def test_regression_performance_report(self):
        result = reports.performance_report(self.df, self.df['y'],
                                        self.df['avg_fair'],
                                        type="regression")

    def test_regression_bias_report(self):
        result = reports.bias_report(self.df, self.df['y'],
                                    self.df['avg_fair'],
                                    type="regression")