# -*- coding: utf-8 -*-
"""
Tools for measuring and comparing fairness across models

Contributors:
    camagallan <christine.allen@kensci.com>
"""
# Copyright (c) KenSci and contributors.
# Licensed under the MIT License.

from abc import ABC
import aif360.sklearn.metrics as aif_mtrc
from collections import *
import fairlearn.metrics as fl_mtrc
from IPython.display import HTML
from joblib import dump, load
import pandas as pd
import numpy as np
import os

from . import reports


# Temporarily hide pandas SettingWithCopy warning
import warnings
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='sklearn')


"""
    Model Comparison Tools
"""


def compare_measures(test_data, target_data, protected_attr_data=None,
                     models=None):
    """ Generates a report comparing fairness measures for the models passed.
            Note: This is a wrapper for the FairCompare.compare_measures method.
            See FairCompare for more information.

        Returns:
            a pandas dataframe
    """
    comp = FairCompare(test_data, target_data, protected_attr_data, models)
    table = comp.compare_measures()
    return(table)


class FairCompare(ABC):
    """ Validates and stores data and models for fairness comparison
    """
    def __init__(self, test_data, target_data, protected_attr_data=None,
                 models=None):
        """
            Args:
                test_data (numpy array or similar pandas object): data to be
                    passed to the models to generate predictions. It's
                    recommended that these be separate data from those used to
                    train the model.
                target_data (numpy array or similar pandas object): target data
                    array corresponding to the test data. It is recommended
                    that the target is not present in the test_data.
                protected_attr_data (numpy array or similar pandas object):
                    data for the protected attributes. These data do not need
                    to be present in test_data, but the rows must correspond
                    with test_data.  Note that values must currently be
                    binary or boolean type.
                models (dict or list-like): the set of trained models to be
                    evaluated. Models can be any object with a scikit-like
                    predict() method. Dict keys assumed as model names. If a
                    list-like object is passed, will set model names relative
                    to their index
        """
        self.X = test_data
        self.protected_attr = protected_attr_data
        self.y = target_data
        self.models = models if models is not None else {}
        try:
            self.__validate()
        except ValidationError as ve:
            raise ValidationError(f"Error loading FairCompare. {ve}")

    def compare_measures(self):
        """ Generates a report that compares fairness measures for all
            entries in self.models

        Returns:
            a pandas dataframe
        """
        self.__validate()
        if len(self.models) == 0:
            warnings.warn("No models to compare.")
            return pd.DataFrame()
        else:
            test_results = []
            # self.measure_model runs __validate, but self has just been
            #   validated. Toggle-off model validation for faster/quieter
            #   processing
            self.__toggle_validation()
            # Compile measure_model results for each model
            for model_name in self.models.keys():
                res = self.measure_model(model_name)
                res.rename(columns={'Value': model_name}, inplace=True)
                test_results.append(res)
            self.__toggle_validation()  # toggle-on model validation
            if len(test_results) > 0:
                output = pd.concat(test_results, axis=1)
                return output
            else:
                return None

    def measure_model(self, model_name):
        """ Generates a report comparing fairness measures for the model_name
                specified

            Returns:
                a pandas dataframe
        """
        self.__validate()
        if model_name not in self.models.keys():
            msg = f"Could not measure fairness for {model_name}. Name" + \
                  f" not found in models. Available models include " + \
                  f" {list(self.models.keys())}"
            logging.info(msg)
            return pd.DataFrame()
        #
        mdl = self.models[model_name]
        X = self.X if not is_dictlike(self.X) else self.X[model_name]
        y = self.y if not is_dictlike(self.y) else self.y[model_name]
        if not is_dictlike(self.protected_attr):
            prtc_attr = self.protected_attr
        else:
            prtc_attr = self.protected_attr[model_name]
        # Verify that predictions can be generated from the test data
        try:
            y_pred = mdl.predict(X)
        except BaseException as e:
            msg = f"Failure generating predictions for {model_name} model." + \
                  " Verify that data are correctly formatted for this model." +\
                  f"{e}"
            raise ValidationError(msg)
        # Since most fairness measures do not require probabilities, y_prob is
        #   optional
        try:
            y_prob = mdl.predict_proba(X)[:, 1]
        except BaseException as e:
            warnings.warn(f"Failure predicting probabilities for {model_name}."
                          + f" Related metrics will be skipped. {e}")
            y_prob = None
        finally:
            res = reports.classification_fairness(X, prtc_attr,
                                                  y, y_pred, y_prob)
            return res

    def save_comparison(self, filepath):
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        dump(self, filepath)

    def __toggle_validation(self):
        if self.__pause_validation:
            self.__pause_validation = False
        else:
            self.__pause_validation = True

    def __validate(self):
        """ Verifies that attributes are set appropriately and updates as
                appropriate

            Raises:
                ValidationError
        """
        # Skip validation if paused
        if self.__validation_paused():
            return None
        #
        valid_data_types = (pd.DataFrame, pd.Series, np.ndarray)
        array_types = (list, tuple, set, dict, OrderedDict)
        dataobj = [self.X, self.y, self.protected_attr]
        validobj = dataobj + [self.models]
        # Do nothing any required objects are missing
        if any(p is None for p in dataobj):
            msg = "Invalid instance: NoneType data arguments are prohibited."
            raise ValidationError(msg)
        # Type and length of X, y, protected_attr must match models if passed
        #   as array-like
        if any(isinstance(p, array_types) for p in dataobj):
            err = None
            if not len(set(type(p) for p in dataobj)) == 1:
                err = "If the data arguments are passed in list/dict," + \
                      " all three must be passed in same list/dict type."
            elif not all(isinstance(p, type(self.models)) for p in dataobj):
                err = "If the data arguments are passed in list/dict" + \
                      " object must be passed as the same type as the" + \
                      " models argument"
            elif not all(len(p) == len(self.models) for p in dataobj):
                err = "If the data arguments are in list-like object, they" + \
                      " must be of same length as the models argument"
            # Comparison function will use keys to iterate in comparisons
            if all(is_dictlike(p) for p in validobj):
                if not all(p.keys() == self.models.keys() for p in dataobj):
                    err = "If the data arguments are passed in dict-like" + \
                          " object, all keys in data arguments must match" + \
                          " the keys in the models argument"
                elif not all(len(p) == len(self.models) for p in dataobj):
                    err = "If the data arguments are passed in list/dict," + \
                          " they must be of same length as the models argument"
            # convert to dict if not already dict. models will be converted
            #    Note: list-like protected_attr and model arguments will be
            #           converted later
            else:
                self.X = {f'model_{i}': m for i, m in enumerate(self.X)}
                self.y = {f'model_{i}': m for i, m in enumerate(self.y)}
            if err is not None:
                raise ValidationError(err)
        ## Validate Test Data (X, y)
        X = {0: self.X} if not is_dictlike(self.X) else self.X
        y = {0: self.y} if not is_dictlike(self.y) else self.y
        for data in [X, y]:
            for d in data.values():
                if not isinstance(d, valid_data_types):
                    msg = "Input data must be numpy array or pandas object," + \
                        " or a list/dict of such objects"
                    raise ValidationError(msg)
        for k in X.keys():
            if not X[k].shape[0] == y[k].shape[0]:
                raise ValidationError("Test and target data mismatch.")
        ## Validate Protected Attributes
        # Ensure that every column of the protected attributes is boolean
        # ToDo: enable prtc_attr as a single multi col array; iterate cols
        if is_dictlike(self.protected_attr):
            prtc_attr = self.protected_attr
        elif isinstance(self.protected_attr, array_types):
            prtc_attr = {f'model_{i}': m for i, m in
                         enumerate(self.protected_attr)}
        else:
            prtc_attr = {0: self.protected_attr}
        #
        for prt_at in prtc_attr.values():
            if not isinstance(prt_at, valid_data_types):
                msg = "Protected attribute(s) must be numpy array or" + \
                    " similar pandas object"
                raise ValidationError(msg)
            data_shape = prt_at.shape
            if len(data_shape) > 1 and data_shape[1] > 1:
                msg = "This library is not yet compatible with" + \
                    " multiple protected attributes."
                raise ValidationError(msg)
        ## Validate Models
        # Ensure models appear as dict
        if not is_dictlike(self.models):
            if not isinstance(self.models, array_types):
                msg = "Models must be dict or list-like group of trained," + \
                      " scikit-like models"
                raise ValidationError(msg)
            self.models = {f'model_{i}': m for i, m in enumerate(self.models)}
            print("Since no model names were passed, the following names have",
                  " been assigned to the models per their indexes:",
                  f" {list(self.models.keys())}")
        # Ensure that models are present and have a predict function
        if self.models is not None:
            if not len(self.models) > 0:
                raise ValidationError("The set of models is empty")
            else:
                for _, m in self.models.items():
                    pred_func = getattr(m, "predict", None)
                    if not callable(pred_func):
                        msg = f"{m} model does not have predict function"
                        raise ValidationError(msg)
        return None

    def __validation_paused(self):
        if not hasattr(self, "__pause_validation"):
            self.__pause_validation = False
        return self.__pause_validation


def is_dictlike(obj):
    dictlike = all([callable(getattr(obj, "keys", None)),
                    not hasattr(obj, "size")])
    return dictlike


def load_comparison(filepath):
    """ Loads a file directly into a FairCompare object

        Returns:
            initialized FairCompare object
    """
    data = load(filepath)
    fair_comp = FairCompare(test_data=data['test_data'],
                            target_data=data['target_data'],
                            models=data['models'],
                            train_data=data['train_data']
                            )
    return fair_comp


class ValidationError(Exception):
    pass


'''
Test Functions
'''


def test_compare():
    rng36 = np.random.default_rng(seed=36)
    rng42 = np.random.default_rng(seed=42)
    X = pd.DataFrame(rng36.integers(0, 1000, size=(100, 4)))
    y = rng36.integers(0, 2, size=(100, 1))
    protected_attr = rng42.integers(0, 2, size=(100, 1))
    models = None
    comparison = compare_measures(X, y, protected_attr, models)
    assert comparison is not None
