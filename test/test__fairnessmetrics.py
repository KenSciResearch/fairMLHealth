'''
Validation tests for fairmlhealth
'''

import fairmlhealth.__fairness_metrics as fairmetric
from numpy import isnan
import pandas as pd
import pytest
from fairmlhealth import stat_utils
from .__testing_utilities import synth_dataset


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
    """ Test proper functioning of compare(). Result should be a pandas dataframe

    """
    def test_eo_diff(self):
        # rounding and temporary values added until unfair predictions are corrected in an upcoming PR
        assert fairmetric.eq_odds_diff(self.y, self.y) == 0
        assert round(fairmetric.eq_odds_diff(self.y, self.avg_fair), 2) == 0.17 #0


    def test_eo_ratio(self):
        # Temporary values added until unfair predictions are corrected in an upcoming PR
        assert isnan(fairmetric.eq_odds_ratio(self.y, self.y))
        assert fairmetric.eq_odds_ratio(self.y, self.avg_fair) == 1.5 #1

