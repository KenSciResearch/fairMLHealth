'''
Validation tests for fairmlhealth
'''

from fairmlhealth.__performance_metrics import (
                                    true_positive_rate, true_negative_rate,
                                    false_negative_rate, false_positive_rate)
import pytest
import numpy as np



class TestClassificationMetrics:
    """ Test proper functioning of the compare_models function. Result
        should be a pandas dataframe

    """
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    prfct_prdctr = y
    avg_prdctr = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    invrs_prdctr = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    def test_tpr(self):
        assert true_positive_rate(self.y, self.prfct_prdctr) == 1
        assert true_positive_rate(self.y, self.avg_prdctr) == 0.5
        assert true_positive_rate(self.y, self.invrs_prdctr) == 0

    def test_fpr(self):
        assert false_positive_rate(self.y, self.prfct_prdctr) == 0
        assert false_positive_rate(self.y, self.avg_prdctr) == 0.5
        assert false_positive_rate(self.y, self.invrs_prdctr) == 1

    def test_tnr(self):
        assert true_negative_rate(self.y, self.prfct_prdctr) == 1
        assert true_negative_rate(self.y, self.avg_prdctr) == 0.5
        assert true_negative_rate(self.y, self.invrs_prdctr) == 0

    def test_fnr(self):
        assert false_negative_rate(self.y, self.prfct_prdctr) == 0
        assert false_negative_rate(self.y, self.avg_prdctr) == 0.5
        assert false_negative_rate(self.y, self.invrs_prdctr) == 1
