# -*- coding: utf-8 -*-
"""
Add-ons for loading data, formatting, and generating tables as part of
KDD 2020 Tutorial on Measuring Fairness for Healthcare.
To be called by Tutorial Notebook.

Contributors:
    camagallen <ca.magallen@gmail.com>
"""
# Copyright (c) KenSci and contributors.
# Licensed under the MIT License.

from .__validation import validate_notebook_requirements
from . import stat_utils, __mimic_data as mimic_data
import pandas as pd
from warnings import warn

'''
Global variable for backward compatibility with KDD2020 tutorial. Used to
    reduce verbosity of comparison tables.
"""
TUTORIAL_ON = False
DeprecationWarning("The KDD2020 tutorial notebook is no longer supported." +
                   " Some portions of this tutorial may no longer function," +
                   " or may have changed since the KDD 2020 design")

def start_tutorial():
    # Don't start the tutorial unless required packages are present
    validate_notebook_requirements()
    global TUTORIAL_ON
    TUTORIAL_ON = True


def stop_tutorial():
    global TUTORIAL_ON
    TUTORIAL_ON = False


def is_kdd_tutorial():
    return TUTORIAL_ON


"""
Formatting Helpers
"""


def highlight_col(df, color="magenta"):
    return f"background-color: {color}"


def highlight_vals(
    df, values, colname=None, criteria=None, color="magenta", h_type="field"
):
    """ Returns a list of strings setting the background color at each index of
        df where a[column] is in the list of values

    Args:
        df (pandas df): any dataframe
        values (list-like): values in colname to be highlighted
        colname (str): name of column against which to match values. Defaults
            to None.
        criteria (str): query criteria. may not . Defaults to None.
        color (str): css color name. Defaults to 'aquamarine'.
        h_type (str, optional):  . Defaults to 'field'.

    Raises:
        ValueError
    """
    if criteria is not None and values is not None:
        print(
            "Cannot process both criteria and values.", "Defaulting to criteria entry"
        )
    if h_type not in ["text", "field"]:
        raise ValueError("Wrong h_type sent")
    if not isinstance(colname, (list, tuple)):
        colname = list(colname)
    if values is None:
        values = []
    if not isinstance(values, (list, tuple)):
        values = list(values)
    #
    if criteria is None:
        criteria = f"in {values}"
    highlight = pd.Series(data=False, index=df.index)
    for col in colname:
        test_vals = values
        if criteria is not None:
            test_vals += df.query(" ".join([col, criteria]))
        highlight[col] = bool(df[col] in values)
    if h_type == "text":
        return [f"color: {color}" if highlight.any() else "" for v in highlight]
    elif h_type == "field":
        return [
            f"background-color: {color}" if highlight.any() else "" for v in highlight
        ]


"""
Loaders and Printers
"""


def load_mimic3_example(mimic_dirpath):
    return mimic_data.load_mimic3_example(mimic_dirpath)

def feature_table(df):
    return stat_utils.feature_table(df)

'''
Tutorial-Specific Helpers
"""


def simplify_tutorial_report(comparison_report_df):
    """Updates a fairness comparison report to exclude FairLearn measures. For
        use in the KDD Tutorial, which first introduces AIF360 measures before
        introducing FairLearn

        Args:
            comparison_report_df (pandas df): a fairMLHealth report
            report

        Returns:
            an updated version of the comparison_report_df
    """
    print(
        "Note: this report has been simplified for this tutorial.",
        "For a more extensive report, omit the simplify_tutorial_report function",
    )
    fl_measures = [
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "equalized_odds_difference",
        "equalized_odds_ratio",
    ]
    ix_vals = comparison_report_df.index
    ix_vals = [v.replace(" ", "_").lower() for v in ix_vals]
    drop_meas = [ix_vals.index(v) for v in ix_vals if v in fl_measures]
    df = comparison_report_df.drop(drop_meas, axis=0)
    return df
