'''
Validation tests for fairmlhealth
'''

from fairmlhealth.__classification_metrics import *
import pytest




class ClassificationMetrics:
    """ Test proper functioning of the compare_models function. Result
        should be a pandas dataframe

    Available functions include accuracy binary_prediction_results
        balanced_accuracy false_negative_rate false_positive_rate f1_score
        negative_predictive_value roc_auc_score precision pr_auc_score precision
        true_negative_rate true_positive_rate
    """
    y = [0,0,0,0,0,0]
    y_avg = [0,1,0,1,0,1]

    def test_accuracy(self):
        accuracy(self.y, self.y_avg)


ClassificationMetrics().test_accuracy()
