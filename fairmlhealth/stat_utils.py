"""
Supplemental functions that may be useful in analysis or data munging.
"""
from numbers import Number
import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Type
from . import __preprocessing as prep, __validation as valid


""" Default P-Value Functions """


def chisquare_pval(
    group: valid.ArrayLike, values: valid.ArrayLike, n_samples: Number = 50
):
    def smpl(ser: pd.Series, n_samp: int = n_samples):
        return ser.sample(n=n_samp, replace=True).reset_index(drop=True)

    #
    g = prep.prep_arraylike(group, "group", expected_len=None)
    v = prep.prep_arraylike(values, "values", expected_len=len(g))
    n = n_samples if n_samples <= g.shape[0] else g.shape[0]
    data = pd.concat([smpl(g, n), smpl(v, n)], axis=1, ignore_index=True)
    data.columns = ["group", "values"]
    # Using groupby and unstack to avoid bugs in some versions of crosstab
    ctab = data.groupby(["group", "values"])["values"].count().unstack().fillna(0)
    pval = stats.chi2_contingency(ctab)[1]
    return pval


def kruskal_pval(a: valid.ArrayLike, b: valid.ArrayLike, n_samples: Number = 50):
    """ Returns the p-value for a Kruskal-Wallis test of a and b using random
        samples of size n_samples.

    Args:
        a (valid.ArrayLike): test distribution
        b (valid.ArrayLike): expected distribution
        n_samples (Number, optional): size of random sample to be tested. if
            fewer than n_samples are present in the data, will set sample size
            to the smaller size between a and b. Defaults to 50.
    """

    def smpl(arr, n_samp):
        return np.random.choice(arr, size=n_samp, replace=True)

    #
    valid.validate_array(a, expected_len=None)
    valid.validate_array(b, expected_len=None)
    min_sample = min(a.shape[0], b.shape[0])
    n = n_samples if n_samples <= min_sample else min_sample
    pval = stats.kruskal(smpl(a, n), smpl(b, n))[1]
    return pval


""" Supplemental Functions """


def binary_result_labels(y_true: valid.ArrayLike, y_pred: valid.ArrayLike):
    y = prep.prep_arraylike(y_true, "y_true", expected_len=None)
    yh = prep.prep_arraylike(y_pred, "y_prob", expected_len=len(y))
    valid.__validate_binVal(y, "y_true", fuzzy=False)
    valid.__validate_binVal(y, "y_pred", fuzzy=False)
    res = pd.Series([""] * len(y_pred), name="prediction result")
    res.loc[y.eq(1) & yh.eq(1)] = "TP"
    res.loc[y.eq(0) & yh.eq(1)] = "FP"
    res.loc[y.eq(0) & yh.eq(0)] = "TN"
    res.loc[y.eq(1) & yh.eq(0)] = "FN"
    return res


def bootstrap_significance(
    func: Callable[[], Number] = kruskal_pval,
    alpha: float = 0.05,
    n_trials: int = 100,
    threshold: float = None,
    **kwargs,
):
    """ Applies bootstrapping to evaluate the frequency of p > alpha. Returns
        True (significant) if the p-value is less than alpha for at least
        P=1-alpha percent of trials.

    Args:
        func (Callable[[kwargs], Number], optional): Any function returning a
            p-value. Defaults to kruskal_pval.
        alpha (float, optional): Maximum p-value indicating significance.
            Defaults to 0.05.
        n_trials (int, optional): Number of trials to run. Defaults to 100.
        threshold (float, optional): minimum proportion of trials exceeding alpha
            required for function to return True. If None, threshold is 1 - alpha.
            Defaults to None.

    Returns:
        bool: whether difference is statistically significant
    """
    if threshold is None:
        threshold = 1 - alpha
    # Create a list of p-values for each of n_trials
    pvals = []
    for i in range(0, n_trials):
        pvals += [func(**kwargs)]
    # Calculate the proportion of trials for which p < alpha
    pvals = [int(v <= alpha) for v in pvals]
    result = bool(np.mean(pvals) >= (threshold))
    return result


def cb_round(series: pd.Series, base: float = 5, sig_dec: int = 0):
    """ Returns the pandas series (or column) with values rounded per the
            custom base value

    Args:
        series (pd.Series): data to be rounded
        base (float): base value to which data should be rounded (may be
            decimal)
        sig_dec (int): number of significant decimals for the
            custom-rounded value
    """
    valid.validate_array(series, "series", expected_len=None)
    if not base >= 0.01:
        err = f"cannot round with base {base}." + "cb_round designed for base >= 0.01."
        raise ValueError(err)
    result = series.apply(lambda x: round(base * round(float(x) / base), sig_dec))
    return result

