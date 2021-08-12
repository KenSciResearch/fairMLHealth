"""
"""
# ToDo: Add more robust testing throughout


from fairmlhealth import measure
import numpy as np
import pytest
import pandas as pd
from .__utils import synth_dataset

np.random.seed(506)


@pytest.fixture(scope="class")
def load_classification_data(request):
    N = 32
    df = synth_dataset(N)
    X = df[["A", "B", "C", "D", "prtc_attr", "other"]]
    y = df["binary_target"].rename("y")
    avg_fair = df["avg_binary_pred"].rename("avg")
    #
    data = pd.concat([X, y, avg_fair], axis=1)
    cohorts = pd.DataFrame(
        {0: np.random.randint(0, 2, N), 1: np.random.randint(0, 2, N)}
    )
    #
    request.cls.df = data
    request.cls.cohorts = cohorts
    yield


@pytest.mark.usefixtures("load_classification_data")
class TestCohorts:
    """ Validates that standard inputs are processed without error
    """

    def test_no_cohort(self):
        _ = measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["y"],
            self.df["avg"],
            pred_type="classification",
        )

    def test_one_cohort(self):
        _ = measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["y"],
            self.df["avg"],
            pred_type="classification",
            cohorts=self.cohorts[0],
        )

    def test_multi_cohort(self):
        _ = measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["y"],
            self.df["avg"],
            pred_type="classification",
            cohorts=self.cohorts,
        )

    def test_cohort_bias(self):
        _ = measure.bias(
            self.df,
            self.df["prtc_attr"],
            self.df["classification"],
            pred_type="classification",
            cohorts=self.cohorts,
        )

    def test_cohort_data(self):
        _ = measure.data(
            self.df, self.df["avg_classification"], cohorts=self.cohorts[0]
        )

    def test_cohort_performance(self):
        _ = measure.performance(
            self.df,
            self.df["classification"],
            self.df["avg_classification"],
            pred_type="classification",
            cohorts=self.cohorts[0],
        )

    def test_cohort_summary(self):
        measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["classification"],
            self.df["avg_classification"],
            cohorts=self.cohorts[0],
            pred_type="classification",
        )


@pytest.mark.usefixtures("load_data")
class TestFlag:
    def test_summary_default_flags_classification(self):
        _ = measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["classification"],
            self.df["avg_classification"],
            pred_type="classification",
            flag_oor=True,
        )

    def test_summary_default_flags_regression(self):
        _ = measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["regression"],
            self.df["avg_regression"],
            pred_type="regression",
            flag_oor=True,
        )

    def test_bias_default_flags_classification(self):
        _ = measure.bias(
            self.df,
            self.df["classification"],
            self.df["avg_classification"],
            pred_type="classification",
            flag_oor=True,
        )

    def test_bias_default_flags_regression(self):
        _ = measure.bias(
            self.df,
            self.df["regression"],
            self.df["avg_regression"],
            pred_type="regression",
            flag_oor=True,
        )

    def test_compare_models_flags(self):
        result = report.compare_models(
            self.df,
            self.df["classification"],
            self.df["prtc_attr"],
            predictions=[self.df["avg_classification"], self.df["avg_classification"]],
            pred_type="classification",
            flag_oor=True,
        )

    def test_flag_with_cohort_summary(self):
        measure.summary(
            self.df,
            self.df["prtc_attr"],
            self.df["classification"],
            self.df["avg_classification"],
            cohorts=self.cohorts[0],
            pred_type="classification",
            flag_oor=True,
        )
