'''
Validation tests for fairmlhealth
'''

import fairmlhealth.__performance_metrics as fpm
import pytest
import numpy as np



class TestClassificationMetrics:
    """
    """
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    prfct_prdctr = y
    avg_prdctr = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    invrs_prdctr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    high_fp = np.array([1, 1, 1, 1, 0, 0, 1, 1]) # low precision
    high_fn = np.array([0, 0, 1, 1, 0, 0, 0, 0]) # zero precision

    def test_bpr(self):
        assert fpm.binary_prediction_results(self.y, self.prfct_prdctr) == \
            {'TP': 4, 'FP': 0, 'TN': 4, 'FN': 0}
        assert fpm.binary_prediction_results(self.y, self.avg_prdctr) == \
            {'TP': 2, 'FP': 2, 'TN': 2, 'FN': 2}
        assert fpm.binary_prediction_results(self.y, self.invrs_prdctr) == \
            {'TP': 0, 'FP': 4, 'TN': 0, 'FN': 4}

    def test_tpr(self):
        assert fpm.true_positive_rate(self.y, self.prfct_prdctr) == 1
        assert fpm.true_positive_rate(self.y, self.avg_prdctr) == 0.5
        assert fpm.true_positive_rate(self.y, self.invrs_prdctr) == 0

    def test_fpr(self):
        assert fpm.false_positive_rate(self.y, self.prfct_prdctr) == 0
        assert fpm.false_positive_rate(self.y, self.avg_prdctr) == 0.5
        assert fpm.false_positive_rate(self.y, self.invrs_prdctr) == 1

    def test_tnr(self):
        assert fpm.true_negative_rate(self.y, self.prfct_prdctr) == 1
        assert fpm.true_negative_rate(self.y, self.avg_prdctr) == 0.5
        assert fpm.true_negative_rate(self.y, self.invrs_prdctr) == 0

    def test_fnr(self):
        assert fpm.false_negative_rate(self.y, self.prfct_prdctr) == 0
        assert fpm.false_negative_rate(self.y, self.avg_prdctr) == 0.5
        assert fpm.false_negative_rate(self.y, self.invrs_prdctr) == 1
