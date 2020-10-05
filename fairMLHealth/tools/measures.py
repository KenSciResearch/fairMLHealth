'''
    Measures of fairness
'''
from aif360.sklearn.metrics import difference, ratio
import numpy as np



def statistical_parity_ratio(y_true, y_pred, prot_attr, weights=None, priv_group=1):
    """ Returns the ratio of the expected values between the privileged and
        unpriveleged group
    """
    def expected_value(pdSeries, weights): return(pdSeries.apply(np.average)[0])
    spr = ratio(expected_value, y_true, y_pred, prot_attr=prot_attr, priv_group=priv_group)
    return(spr)

