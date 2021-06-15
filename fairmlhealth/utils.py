'''
Back-end functions used throughout the library
'''
from importlib.util import find_spec
from .__validation import ValidationError


def cb_round(series, base=5, sig_dec=0):
    """ Returns the pandas series (or column) with values rounded per the
            custom base value

        Args:
            series (pd.Series): data to be rounded
            base (float): base value to which data should be rounded (may be
                decimal)
            sig_dec (int): number of significant decimals for the
                custom-rounded value
    """
    if not base >= 0.01:
        err = (f"cannot round with base {base}." +
               "cb_round designed for base >= 0.01.")
        raise ValueError(err)
    result = series.apply(lambda x: round(base * round(float(x)/base), sig_dec))
    return result


def is_dictlike(obj):
    dictlike = all([callable(getattr(obj, "keys", None)),
                    not hasattr(obj, "size")])
    return dictlike


def validate_notebook_requirements():
    """ Alerts the user if they're missing packages required to run extended
        tutorial and example notebooks
    """
    if find_spec('fairlearn') is None:
        err = ("This notebook cannot be re-run witout Fairlearn, available " +
               "via https://github.com/fairlearn/fairlearn. Please install " +
               "Fairlearn to run this notebook.")
        raise ValidationError(err)
    else:
        pass

