# -*- coding: utf-8 -*-
"""
Tools for measuring and comparing fairness across models

Contributors:
    camagallan <christine.allen@kensci.com>
"""
# Copyright (c) KenSci and contributors.
# Licensed under the MIT License.

from abc import ABC
from collections import OrderedDict
from joblib import dump
import logging
import numpy as np
import pandas as pd
import os

from fairmlhealth.utils import is_dictlike
from fairmlhealth import reports


# Temporarily hide pandas SettingWithCopy warning
import warnings
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='sklearn')


"""
    Model Comparison Tools
"""


def measure_model(test_data, target_data, protected_attr_data, model):
    """ Generates a report of fairness measures for the model

    Args:
        test_data ([type]): [description]
        target_data ([type]): [description]
        protected_attr_data ([type]): [description]
        model ([type]): [description]
    """
    comp = FairCompare(test_data, target_data, protected_attr_data, [model],
                       verboseMode=False)
    table = comp.measure_model(comp.models.keys()[0]).transpose()
    return table


def compare_models(test_data, target_data, protected_attr_data, models):
    """ Generates a report of fairness measures for the model

    Args:
        test_data ([type]): [description]
        target_data ([type]): [description]
        protected_attr_data ([type]): [description]
        model ([type]): [description]
    """
    comp = FairCompare(test_data, target_data, protected_attr_data, models,
                       verboseMode=False)
    table = comp.compare_measures().transpose()
    return table


def compare_measures(test_data, target_data, protected_attr_data=None,
                     models=None):
    """ Generates a report comparing fairness measures for the models passed.
            Note: This is a wrapper for the FairCompare.compare_measures method
            See FairCompare for more information.

    Returns:
        pandas dataframe of fairness and performance measures for each model
    """
    comp = FairCompare(test_data, target_data, protected_attr_data, models,
                       verboseMode=False)
    table = comp.compare_measures()
    return table


class FairCompare(ABC):
    """ Validates and stores data and models for fairness comparison
    """
    def __init__(self, test_data, target_data, protected_attr_data=None,
                 models=None, **kwargs):
        """ Generates fairness comparisons

        Args:
            test_data (numpy array or similar pandas object): data to be
                passed to the models to generate predictions. It's
                recommended that these be separate data from those used to
                train the model.
            target_data (numpy array or similar pandas object): target data
                array corresponding to the test data. It is recommended that
                the target is not present in the test_data.
            protected_attr_data (numpy array or similar pandas object):
                data for the protected attributes. These data do not need to
                be present in test_data, but the rows must correspond
                with test_data.  Note that values must currently be
                binary or boolean type.
            models (dict or list-like): the set of trained models to be
                evaluated. Models can be any object with a scikit-like
                predict() method. Dict keys assumed as model names. If a
                list-like object is passed, will set model names relative to
                their index
        """
        self.X = test_data
        self.protected_attr = protected_attr_data
        self.y = target_data
        self.models = models if models is not None else {}
        self.verboseMode = True
        if "verboseMode" in kwargs:
            self.verboseMode = kwargs.get("verboseMode")
        try:
            self.__validate()
        except ValidationError as ve:
            raise ValidationError(f"Error loading FairCompare. {ve}")

    def compare_measures(self):
        """ Returns a pandas dataframe containing fairness and performance
            measures for all available models
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
        """ Returns a pandas dataframe containing fairness measures for the
                model_name specified

        Args:
            model_name (str): a key corresponding to the model of interest,
                as found in the object's "models" dictionary
        """
        self.__validate()
        if model_name not in self.models.keys():
            msg = (f"Could not measure fairness for {model_name}. Name"
                   f" not found in models. Available models include "
                   f" {list(self.models.keys())}")
            print(msg)
            return pd.DataFrame()
        # Subset to objects for this specific model
        mdl = self.models[model_name]
        X = self.X if not is_dictlike(self.X) else self.X[model_name]
        y = self.y if not is_dictlike(self.y) else self.y[model_name]
        if not is_dictlike(self.protected_attr):
            prtc_attr = self.protected_attr
        else:
            prtc_attr = self.protected_attr[model_name]
        # Verify that predictions can be generated from the test data
        if mdl is None:
            print("No model defined")
            return None
        try:
            y_pred = mdl.predict(X)
        except BaseException as e:
            msg = (f"Failure generating predictions for {model_name} model."
                   " Verify if data are correctly formatted for this model."
                   f"{e}")
            raise ValidationError(msg)
        # Since most fairness measures do not require probabilities, y_prob is
        #   optional
        try:
            y_prob = mdl.predict_proba(X)[:, 1]
        except BaseException as e:
            msg = (f"Failure predicting probabilities for {model_name}."
                   f" Related metrics will be skipped. {e}\n")
            warnings.warn(msg)
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
        self.__pause_validation = not self.__pause_validation

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
                err = ("If the data arguments are passed in list/dict,"
                       " all three must be passed in same list/dict type.")
            elif not all(isinstance(p, type(self.models)) for p in dataobj):
                err = ("If the data arguments are passed in list/dict"
                       " object must be passed as the same type as the"
                       " models argument")
            elif not all(len(p) == len(self.models) for p in dataobj):
                err = ("If the data arguments are in list-like object, they"
                       " must be of same length as the models argument")

            # Comparison function will use keys to iterate in comparisons
            if all(is_dictlike(p) for p in validobj):
                if not all(p.keys() == self.models.keys() for p in dataobj):
                    err = ("If the data arguments are passed in dict-like"
                           " object, all keys in data arguments must match"
                           " the keys in the models argument")
                elif not all(len(p) == len(self.models) for p in dataobj):
                    err = ("If the data arguments are passed in list/dict,"
                           " the list/dict must be the same length as the"
                           " models argument")

            # convert to dict if not already dict. models will be converted
            #    Note: list-like protected_attr and model arguments will be
            #           converted later
            else:
                self.X = {f'model_{i}': m for i, m in enumerate(self.X)}
                self.y = {f'model_{i}': m for i, m in enumerate(self.y)}
            if err is not None:
                raise ValidationError(err)

        # Validate Test Data
        Xd = {0: self.X} if not is_dictlike(self.X) else self.X
        yd = {0: self.y} if not is_dictlike(self.y) else self.y
        for data in [Xd, yd]:
            for d in data.values():
                if not isinstance(d, valid_data_types):
                    msg = ("Input data must be numpy array or pandas object,"
                           " or a list/dict of such objects")
                    raise ValidationError(msg)
        for k in Xd.keys():
            if not Xd[k].shape[0] == yd[k].shape[0]:
                raise ValidationError("Test and target data mismatch.")
        ## Validate Protected Attributes
        # Ensure that every column of the protected attributes is boolean
        # ToDo: enable prtc_attr as a single multi col array; iterate cols
        if is_dictlike(self.protected_attr):
            prtc_attr = self.protected_attr
        elif isinstance(self.protected_attr, array_types):
            self.protected_attr = {f'model_{i}': m for i, m in
                                   enumerate(self.protected_attr)}
            prtc_attr = self.protected_attr
        else:
            prtc_attr = {0: self.protected_attr}
        #
        if prtc_attr.nunique() < 2:
            msg = "Single label found in protected attribute (2 expected)."
            raise ValidationError(msg)
        elif prtc_attr.nunique() > 2:
            msg = ("Multiple labels found in protected attribute"
                    "(2 expected).")
            raise ValidationError(msg)
        for prt_at in prtc_attr.values():
            if not isinstance(prt_at, valid_data_types):
                msg = ("Protected attribute(s) must be numpy array or"
                       " similar pandas object")
                raise ValidationError(msg)
            data_shape = prt_at.shape
            if len(data_shape) > 1 and data_shape[1] > 1:
                msg = ("This library is not yet compatible with groups of"
                       " protected attributes.")
                raise ValidationError(msg)
        ## Validate Models
        # Ensure models appear as dict
        if not is_dictlike(self.models) and self.models is not None:
            if not isinstance(self.models, array_types):
                self.models =[self.models]
            self.models = {f'model_{i}': m for i, m in enumerate(self.models)}
            if self.verboseMode:
                print("Since no model names were passed, the following names have",
                  " been assigned to the models per their indexes:",
                  f" {list(self.models.keys())}")

        # Ensure that models are present and have a predict function
        if self.models is not None:
            if not len(self.models) > 0:
                msg = "Cannot generate comparison with an empty set of models."
                logging.warning(msg)
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


class ValidationError(Exception):
    pass
