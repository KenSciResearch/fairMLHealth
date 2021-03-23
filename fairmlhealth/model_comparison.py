# -*- coding: utf-8 -*-
"""
Tools for measuring and comparing fairness across models

Contributors:
    camagallan <ca.magallen@gmail.com>
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
import warnings

from fairmlhealth.utils import is_dictlike
from fairmlhealth import reports



# Temporarily hide pandas SettingWithCopy warning
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
    model_name = list(comp.models.keys())[0]
    table = comp.measure_model(model_name, skip_performance=True)
    return table


def compare_models(test_data, target_data, protected_attr_data, models):
    """ Generates a report comparing fairness measures for the models passed.
            Note: This is a wrapper for the FairCompare.compare_measures method
            See FairCompare for more information.

    Returns:
        pandas dataframe of fairness and performance measures for each model
    """
    comp = FairCompare(test_data, target_data, protected_attr_data, models,
                       verboseMode=True)
    table = comp.compare_measures()
    return table


def compare_measures(test_data, target_data, protected_attr_data, models):
    """ Generates a report comparing fairness measures for the models passed.
            Note: This is a wrapper for the FairCompare.compare_measures method
            See FairCompare for more information.

    Returns:
        pandas dataframe of fairness and performance measures for each model
    """
    warnings.warn(
            "compare_measures will be deprecated in version 2." +
            " Use compare_models instead.", PendingDeprecationWarning
        )
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

        self.__data_types = (pd.DataFrame, pd.Series, np.ndarray)
        self.__array_types = (list, tuple, set, dict, OrderedDict)
        self.__dataobjects = lambda: [self.X, self.y, self.protected_attr]

        if "verboseMode" in kwargs:
            self.verboseMode = kwargs.get("verboseMode")
        else:
            self.verboseMode = True

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

    def measure_model(self, model_name, **kwargs):
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
                                                  y, y_pred, y_prob, **kwargs)
            return res

    def save_comparison(self, filepath):
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        dump(self, filepath)

    def __paused_validation(self):
        if not hasattr(self, "__pause_validation"):
            self.__pause_validation = False
        return self.__pause_validation

    def __toggle_validation(self):
        self.__pause_validation = not self.__pause_validation

    def __validate(self):
        """ Verifies that attributes are set appropriately and updates as
                appropriate

        Raises:
            ValidationError
        """
        # Skip validation if paused
        if self.__paused_validation():
            return None

        self.__validate_models()
        self.__validate_data()
        return None

    def __validate_data(self):
        """ Verifies that data are of correct type and length; stores objects
            in dicts if not already in them

        Raises:
            ValidationError
        """
        #ToDo: rename - this both validates and updates
        # Models must be dict to run this function
        if not is_dictlike(self.models):
            self.__validate_models()

        # Validate types
        for p in self.__dataobjects():
            err = None
            if p is None:
                err = "NoneType data arguments are prohibited."
            elif (not isinstance(p, self.__data_types) and
                    not len(p) == len(self.models)):
                err = ("If the data arguments are in list-like, they"
                       " must be of same length as the models argument")
            elif is_dictlike(self.models):
                if (is_dictlike(p) and
                    not set(p.keys()) == set(self.models.keys())):
                    err = ("If the data arguments are passed in dict-like"
                           " object, all keys in data arguments must match"
                           " the keys in the models argument")
            if err is not None:
                raise ValidationError(err)

        ''' Update all data to dictionary objects '''
        # Set X as dict if not already
        if isinstance(self.X, self.__data_types):
            self.X = [self.X for i, m in enumerate(self.models)]
        if not is_dictlike(self.X):
            X = self.X
            self.X = {k: X[i] for i, k in enumerate(self.models.keys())}
        # Set y as dict if not already
        if isinstance(self.y, self.__data_types):
            self.y = [self.y for i, m in enumerate(self.models)]
        if not is_dictlike(self.y):
            y = self.y
            self.y = {k: y[i] for i, k in enumerate(self.models.keys())}
        # Set protected_attr as dict if not already
        if isinstance(self.protected_attr, self.__data_types):
            self.protected_attr = [self.protected_attr
                                   for i, m in enumerate(self.models)]
        if not is_dictlike(self.protected_attr):
            p = self.protected_attr
            self.protected_attr = {k: p[i]
                                   for i, k in enumerate(self.models.keys())}

        # Validate Entries
        for data in self.__dataobjects():
            for d in data.values():
                if not isinstance(d, self.__data_types):
                    err = ("Input data must be numpy array or pandas object,"
                           " or a list/dict of such objects")
                    raise ValidationError(err)
        for k in self.X.keys():
            if not self.X[k].shape[0] == self.y[k].shape[0]:
                raise ValidationError("Test and target data mismatch.")
            if not self.X[k].shape[0] == self.protected_attr[k].shape[0]:
                err = "Test data and protected attribute mismatch."
                raise ValidationError(err)

        # Ensure that protected attribute is binary- or boolean-type
        for _, prt_at in self.protected_attr.items():
            err = None
            if not isinstance(prt_at, self.__data_types):
                err = ("Protected attribute(s) must be numpy array or"
                       " similar pandas object")
            if len(prt_at.shape) > 1 and prt_at.shape[1] > 1:
                err = ("This library is not yet compatible with groups of"
                       " protected attributes.")
            if len(np.unique(prt_at)) < 2:
                err = "Single label found in protected attribute (2 expected)."
            elif len(np.unique(prt_at)) > 2:
                err = ("Multiple labels found in protected attribute"
                        "(only 2 allowed).")
            if err is not None:
                raise ValidationError(err)

    def __validate_models(self):
        """ Verifies that models have predict function and stores them in a dict
            if not already in one

        Raises:
            ValidationError
        """
        #ToDo: rename - this both validates and updates
        # Set models as dict if not already
        if not is_dictlike(self.models) and self.models is not None:
            if not isinstance(self.models, self.__array_types):
                self.models = [self.models]
            self.models = {f'model_{i}': m for i, m in enumerate(self.models)}
            if self.verboseMode:
                print("Since no model names were passed, the following names",
                      "have been assigned to the models per their indexes:",
                      f" {list(self.models.keys())}")

        # Ensure that model dict contains model objects posessing a predict
        #   function
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


class ValidationError(Exception):
    pass
