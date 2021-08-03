'''
'''


from logging import warning
from fairmlhealth import measure, report, __validation as valid
import numpy as np
import pytest
import pandas as pd
from .__testing_utilities import synth_dataset
np.random.seed(506)


@pytest.fixture(scope="class")
def load_data(request):
    N = 32
    df = synth_dataset(N)
    X = df[['A', 'B', 'C', 'D', 'prtc_attr', 'other']]
    y_b = df['binary_target'].rename('classification')
    avg_fair_b = df['avg_binary_pred'].rename('avg_classification')
    y_r = df['continuous_target'].rename('regression')
    avg_fair_r = df['avg_cont_pred'].rename('avg_regression')
    #
    data = pd.concat([X, y_b, avg_fair_b, y_r, avg_fair_r], axis=1)
    cohorts = pd.DataFrame({0:np.random.randint(0, 2, N),
                            1:np.random.randint(0, 2, N)}
                           )
    #
    request.cls.df = data
    request.cls.cohorts = cohorts
    yield


@pytest.mark.usefixtures("load_data")
class TestCohorts:
    """ Validates that standard inputs are processed without error
    """
    def test_no_cohort(self):
        _ = measure.summary(self.df, self.df['prtc_attr'],
                            self.df['classification'],
                            self.df['avg_classification'],
                            pred_type="classification")

    def test_one_cohort_cols(self):
        _ = measure.summary(self.df, self.df['prtc_attr'],
                            self.df['classification'],
                            self.df['avg_classification'],
                            pred_type="classification",
                            cohorts=self.cohorts[0])

    def test_multi_cohort_cols(self):
        _ = measure.summary(self.df, self.df['prtc_attr'],
                            self.df['classification'],
                            self.df['avg_classification'],
                            pred_type="classification",
                            cohorts=self.cohorts)

    def test_toomany_cohorts(self):
        tmc = self.df['A'].reset_index()
        with pytest.raises(valid.ValidationError):
            _ = measure.summary(self.df, self.df['prtc_attr'],
                                self.df['classification'],
                                self.df['avg_classification'],
                                pred_type="classification",
                                cohorts=tmc['index'])

    def test_cohort_bias(self):
        _ = measure.bias(self.df, self.df['prtc_attr'],
                            self.df['classification'],
                            pred_type="classification",
                            cohorts=self.cohorts)

    def test_cohort_data(self):
        _ = measure.data(self.df,
                         self.df['avg_classification'],
                         cohorts=self.cohorts[0])

    def test_cohort_performance(self):
        _ = measure.performance(self.df,
                                self.df['classification'],
                                self.df['avg_classification'],
                                pred_type="classification",
                                cohorts=self.cohorts[0])

    def test_cohort_summary(self):
        measure.summary(self.df,
                        self.df['prtc_attr'],
                        self.df['classification'],
                        self.df['avg_classification'],
                        cohorts=self.cohorts[0],
                        pred_type="classification")


@pytest.mark.usefixtures("load_data")
class TestFlag():
    def test_summary_default_flags_classification(self):
        _ = measure.summary(self.df,
                            self.df['prtc_attr'],
                            self.df['classification'],
                            self.df['avg_classification'],
                            pred_type="classification",
                            flag_oor=True)

    def test_summary_default_flags_regression(self):
        _ = measure.summary(self.df,
                            self.df['prtc_attr'],
                            self.df['regression'],
                            self.df['avg_regression'],
                            pred_type="regression",
                            flag_oor=True)

    def test_bias_default_flags_classification(self):
        _ = measure.bias(self.df,
                         self.df['classification'],
                         self.df['avg_classification'],
                         pred_type="classification",
                         flag_oor=True)

    def test_bias_default_flags_regression(self):
        _ = measure.bias(self.df,
                            self.df['regression'],
                            self.df['avg_regression'],
                            pred_type="regression",
                            flag_oor=True)

    def test_compare_flags(self):
        result = report.compare(self.df,
                                       self.df['classification'],
                                       self.df['prtc_attr'],
                                       predictions=[self.df['avg_classification'],
                                                    self.df['avg_classification']
                                                    ],
                                        pred_type="classification",
                                        flag_oor=True)

    def test_flag_with_cohort_summary(self):
        measure.summary(self.df,
                        self.df['prtc_attr'],
                        self.df['classification'],
                        self.df['avg_classification'],
                        cohorts=self.cohorts[0],
                        pred_type="classification",
                        flag_oor=True)
