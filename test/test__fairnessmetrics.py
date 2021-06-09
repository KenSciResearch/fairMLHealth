'''
Validation tests for fairmlhealth
'''

from fairmlhealth.__fairness_metrics import eq_odds_diff, eq_odds_ratio
import pandas as pd
import pytest



class TestFairnessMetrics:
    """ Test proper functioning of the compare_models function. Result
        should be a pandas dataframe

    """
    pa = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="protected_attribute")
    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1], index=pa, name="y")
    prfct_fair = y
    avg_fair = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=pa, name="avg")
    bias_againstupg = pd.Series([1, 1, 1, 0, 0, 1, 1, 1], index=pa, name="anti")
    bias_forupg = pd.Series([0, 1, 1, 1, 1, 1, 0, 1], index=pa, name="pro")


    def test_eo_diff(self):
        assert eq_odds_diff(self.y, self.prfct_fair) == 0
        assert eq_odds_diff(self.y, self.avg_fair) == 0
        assert eq_odds_diff(self.y, self.bias_againstupg) == -0.5
        assert eq_odds_diff(self.y, self.bias_forupg) == 0.5

    def test_eo_ratio(self):
        # The perfect predictor raises a warning d.t. no false positives
        assert eq_odds_ratio(self.y, self.prfct_fair) == 0
        assert eq_odds_ratio(self.y, self.avg_fair) == 1
        assert eq_odds_ratio(self.y, self.bias_againstupg) == 2
        assert eq_odds_ratio(self.y, self.bias_forupg) == 2
