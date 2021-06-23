'''
Validation tests for fairmlhealth
'''

from fairmlhealth.__fairness_metrics import eq_odds_diff, eq_odds_ratio
from numpy import isclose
import pandas as pd
import pytest
from .__utils import synth_dataset


@pytest.fixture(scope="class")
def load_data(request):
    df = synth_dataset(24)
    pa = pd.Series(df['prtc_attr'])
    request.cls.pa = pa
    request.cls.y = df[['binary_target']].set_index(pa)
    request.cls.avg_fair = df[['avg_binary_pred']].set_index(pa)
    request.cls.bias_against0 = df[['binary_bias_against0']].set_index(pa)
    request.cls.bias_toward0 = df[['binary_bias_toward0']].set_index(pa)
    yield


@pytest.mark.usefixtures("load_data")
class TestFairnessMetrics:
    """ Test proper functioning of the compare_models function. Result
        should be a pandas dataframe

    """
    def test_eo_diff(self):
        # rounding and temporary values added until unfair predictions are corrected in an upcoming PR
        assert eq_odds_diff(self.y, self.y) == 0
        assert round(eq_odds_diff(self.y, self.avg_fair), 2) == 0.17 #0
        assert eq_odds_diff(self.y, self.bias_against0) == -0.5
        assert round(eq_odds_diff(self.y, self.bias_toward0), 2) == 0.67 #0.5

    def test_eo_ratio(self):
        # Temporary values added until unfair predictions are corrected in an upcoming PR
        assert eq_odds_ratio(self.y, self.y) == 0
        assert eq_odds_ratio(self.y, self.avg_fair) == 1.5 #1
        assert eq_odds_ratio(self.y, self.bias_against0) == 0
        assert eq_odds_ratio(self.y, self.bias_toward0) == 3 #2
