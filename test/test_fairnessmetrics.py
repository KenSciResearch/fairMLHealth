"""
Validation tests for fairmlhealth
"""

import fairmlhealth.fairness_metrics as fm
from numpy import isnan
import pandas as pd
import pytest
from fairmlhealth import stat_utils
from .__utils import synth_dataset


@pytest.fixture(scope="class")
def load_data(request):
    df = synth_dataset(24)
    prtc = pd.Series(df["prtc_attr"])
    request.cls.pa = "prtc_attr"
    request.cls.y = df[["binary_target"]].set_index(prtc)
    request.cls.avg_fair = df[["avg_binary_pred"]].set_index(prtc)
    request.cls.bias_against0 = df[["binary_bias_against0"]].set_index(prtc)
    request.cls.bias_toward0 = df[["binary_bias_toward0"]].set_index(prtc)
    yield


@pytest.mark.usefixtures("load_data")
class TestFairnessMetrics:
    """ Test proper functioning of compare(). Result should be a pandas dataframe

    """

    def test_fpr_diff(self):
        assert fm.fpr_diff(self.y, self.y, self.pa) == 0
        assert round(fm.fpr_diff(self.y, self.avg_fair, self.pa), 2) == 0.06
        assert round(fm.fpr_diff(self.y, self.bias_against0, self.pa), 2) == 0.83
        assert round(fm.fpr_diff(self.y, self.bias_toward0, self.pa), 2) == -0.89

    def test_tpr_diff(self):
        assert fm.tpr_diff(self.y, self.y, self.pa) == 0
        assert round(fm.tpr_diff(self.y, self.avg_fair, self.pa), 2) == 0.17
        assert round(fm.tpr_diff(self.y, self.bias_against0, self.pa), 2) == -0.67
        assert round(fm.tpr_diff(self.y, self.bias_toward0, self.pa), 2) == 0.67

    def test_ppv_diff(self):
        assert fm.ppv_diff(self.y, self.y, self.pa) == 0
        assert fm.ppv_diff(self.y, self.avg_fair, self.pa) == 0.3
        assert round(fm.ppv_diff(self.y, self.bias_against0, self.pa), 2) == -0.71
        assert round(fm.ppv_diff(self.y, self.bias_toward0, self.pa), 2) == 0.89

    def test_fpr_ratio(self):
        assert round(fm.fpr_ratio(self.y, self.avg_fair, self.pa), 2) == 1.12
        assert isnan(fm.fpr_ratio(self.y, self.bias_against0, self.pa))
        assert fm.fpr_ratio(self.y, self.bias_toward0, self.pa) == 0

    def test_tpr_ratio(self):
        assert fm.tpr_ratio(self.y, self.y, self.pa) == 1
        assert fm.tpr_ratio(self.y, self.avg_fair, self.pa) == 1.5
        assert round(fm.tpr_ratio(self.y, self.bias_against0, self.pa), 2) == 0.33
        assert fm.tpr_ratio(self.y, self.bias_toward0, self.pa) == 3

    def test_ppv_ratio(self):
        assert fm.ppv_ratio(self.y, self.y, self.pa) == 1
        assert fm.ppv_ratio(self.y, self.avg_fair, self.pa) == 2.5
        assert round(fm.ppv_ratio(self.y, self.bias_against0, self.pa), 2) == 0.29
        assert fm.ppv_ratio(self.y, self.bias_toward0, self.pa) == 9

    def test_eo_diff(self):
        # rounding and temporary values added until unfair predictions are corrected in an upcoming PR
        assert fm.eq_odds_diff(self.y, self.y, self.pa) == 0
        assert round(fm.eq_odds_diff(self.y, self.avg_fair, self.pa), 2) == 0.17  # 0

    def test_eo_ratio(self):
        # Temporary values added until unfair predictions are corrected in an upcoming PR
        assert isnan(fm.eq_odds_ratio(self.y, self.y, self.pa))
        assert fm.eq_odds_ratio(self.y, self.avg_fair, self.pa) == 1.5  # 1
