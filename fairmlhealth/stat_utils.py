'''
Supplemental functions helpful in supporting analysis
'''
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from typing import Callable
from . import __validation as valid


def bootstrap_significance(a, b, func:Callable=kruskal, alpha:float=0.05,
                           n_samples:int=50, n_trials:int=100):
    """ Applies bootstrapping to evaluate the statistical difference between
    two samples. Returns True (significant) if the p-value is less than alpha
    for at least P=1-alpha percent of trials.

    Args:
        a (array-like): statistical sample.
        b (array-like): statistical sample for comparison.
        func (function): any statistical test returning it's p-value as the
            second member of a tuple.
        alpha (float, optional): Maximum p-value indicating significance.
            Defaults to 0.05.
        n_samples (int, optional): Number of samples to use for each trial.
            Defaults to 50.
        n_trials (int, optional): Number of trials to run. Defaults to 100.

    Returns:
        bool: whether difference is statistically significant
    """
    valid.validate_array(a, "a", expected_len=None)
    valid.validate_array(b, "b", expected_len=len(a))
    # Create a list of p-values for each of n_trials
    pvals = []
    for i in range( 0, n_trials):
        pvals += [func(np.random.choice(a, size=n_samples, replace=True),
                       np.random.choice(b, size=n_samples, replace=True))[1]
                  ]
    # Calculate the proportion of trials for which p < alpha
    pvals = [int(v <= alpha) for v in pvals]
    result = bool(np.mean(pvals) >= (1-alpha))
    return result


def cb_round(series:pd.Series, base:float=5, sig_dec:int=0):
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
        err = (f"cannot round with base {base}." +
               "cb_round designed for base >= 0.01.")
        raise ValueError(err)
    result = series.apply(lambda x: round(base * round(float(x)/base), sig_dec))
    return result


def feature_table(df:pd.DataFrame):
    ''' Displays a table containing statistics on the features available in the
            passed df

        Args:
            df (pandas df): dataframe containing MIMIC data for the tutorial
    '''
    if df is None or not isinstance(df, pd.DataFrame):
        raise valid.ValidationError(
            "feature_table is designed for pandas DataFrame objects only")
    print(f"\n This data subset has {df.shape[0]} total observations" +
          f" and {df.shape[1]-2} input features \n")
    feat_df = pd.DataFrame({'feature': df.columns.tolist()
                            }).query(
                                'feature not in ["ADMIT_ID", "length_of_stay"]')
    feat_df['Raw Feature'] = feat_df['feature'].str.split("_").str[0]
    count_df = feat_df.groupby('Raw Feature', as_index=False
                               )['feature'].count(
                     ).rename(columns={
                              'feature': 'Category Count (Encoded Features).'})
    return count_df

