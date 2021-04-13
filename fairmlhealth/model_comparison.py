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
from fairmlhealth.reports import classification_fairness as classfair



# Temporarily hide pandas SettingWithCopy warning
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='sklearn')


"""
    Model Comparison Tools
"""


def measure_model(test_data, targets, protected_attr, model=None,
                  predictions=None, probabilities=None):
    """ Generates a report of fairness measures for the model

    Args:
        test_data (pandas DataFrame or compatible type): [description]
        targets (pandas Series or compatible type): [description]
        protected_attr (pandas Series or compatible type): [description]
        model (scikit model or other model object with a *.predict() function
            that accepts test_data and returns an array of predictions).
            Defaults to None. If None, must pass predictions.
        predictions (pandas Series or compatible type): Set of predictions
            corresponding to targets. Defaults to None. Ignored
            if model argument is passed.
        probabilities (pandas Series or compatible type): Set of probabilities
            corresponding to predictions. Defaults to None. Ignored
            if models argument is passed.

    Returns:
        pandas dataframe of fairness measures for the model
    """
    comp = FairCompare(test_data, targets, protected_attr, [model],
                       predictions, probabilities, verboseMode=False)
    model_name = list(comp.models.keys())[0]
    table = comp.measure_model(model_name, skip_performance=True)
    return table


def compare_models(test_data, target_data, protected_attr, models=None,
                   predictions=None):
    """ Generates a report comparing fairness measures for the models passed.
            Note: This is a wrapper for the FairCompare.compare_measures method
            See FairCompare for more information.

    Args:
        test_data (pandas DataFrame or compatible type): [description]
        targets (pandas Series or compatible type): [description]
        protected_attr (pandas Series or compatible type): [description]
        model (scikit model or other model object with a *.predict() function
            that accepts test_data and returns an array of predictions).
            Defaults to None. If None, must pass predictions.
        predictions (pandas Series or compatible type): Set of predictions
            corresponding to targets. Defaults to None. Ignored
            if model argument is passed.
        probabilities (pandas Series or compatible type): Set of probabilities
            corresponding to predictions. Defaults to None. Ignored
            if models argument is passed.

    Returns:
        pandas dataframe of fairness and performance measures for each model
    """
    comp = FairCompare(test_data, target_data, protected_attr, models,
                       predictions, verboseMode=True)
    table = comp.compare_measures()
    return table


def compare_measures(test_data, target_data, protected_attr_data, models=None,
                     predictions=None):
    """ Deprecated in favor of compare_models. Generates a report comparing
        fairness measures for the models passed.Note: This is a wrapper for the
        FairCompare.compare_measures method See FairCompare for more
        information.

    Returns:
        pandas dataframe of fairness and performance measures for each model
    """
    warnings.warn(
            "compare_measures will be deprecated in version 2." +
            " Use compare_models instead.", PendingDeprecationWarning
        )
    comp = FairCompare(test_data, target_data, protected_attr_data, models,
                       predictions, verboseMode=False)
    table = comp.compare_measures()
    return table


class FairCompare(ABC):
    """ Validates and stores data and models for fairness comparison
    """

    def __init__(self, test_data, target_data, protected_attr=None,
                 models=None, preds=None, probs=None, **kwargs):
        """ Generates fairness comparisons

        Args:
            test_data (numpy array or similar pandas object): data to be
                passed to the models to generate predictions. It's
                recommended that these be separate data from those used to
                train the model.
            target_data (numpy array or similar pandas object): target data
                array corresponding to the test data. It is recommended that
                the target is not present in the test_data.
            protected_attr (numpy array or similar pandas object):
                data for the protected attributes. These data do not need to
                be present in test_data, but the rows must correspond
                with test_data.  Note that values must currently be
                binary or boolean type.
            models (dict or list-like): the set of trained models to be
                evaluated. Models can be any object with a scikit-like
                predict() method. Dict keys assumed as model names. If a
                list-like object is passed, will set model names relative to
                their index
            preds (pandas Series or compatible type): Set of predictions
                corresponding to targets. Defaults to None. Ignored
                if model argument is passed.
            probs (pandas Series or compatible type): Set of probabilities
                corresponding to predictions. Defaults to None. Ignored
                if models argument is passed.
        """
        #
        self.X = test_data
        self.prtc_attr = protected_attr
        self.y = target_data

        # The user is forced to pass either models or predictions as None to
        # simplify attribute management. If models are passed, they will be used
        # to produce predictions.
        if not (models is None or preds is None):
            err = ("FairCompare accepts either models or predictions, but not"
                   + " both")
            raise ValidationError(err)
        self.models = models if models not in [None, [None]] else None
        self.preds = None if self.models is not None else preds
        self.probs = None if self.models is not None else probs

        #
        self.__meas_obj = ["X", "y", "prtc_attr", "models", "preds", "probs"]
        self.__data_types = (pd.DataFrame, pd.Series, np.ndarray)
        self.__iter_types = (list, tuple, set, dict, OrderedDict)
        #
        if "verboseMode" in kwargs:
            self.verboseMode = kwargs.get("verboseMode")
        else:
            self.verboseMode = True
        #
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
        res = classfair(self.X[model_name], self.prtc_attr[model_name],
                        self.y[model_name], self.preds[model_name],
                        self.probs[model_name], **kwargs)
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

    def __set_dicts(self):
        """ Ensures correct datatypes for model measurement, including the
            following actions:
                - stores all measure-relevant properties as dictionaries with
                model_names as keys.
                - ensures that each dictionary entry is of a type that can
                be measured by this tool
        """
        # Iterable attributes must be of same length so that keys can be
        # properly matched when they're converted to dictionaries.
        expected_len = 1 # expected len of iterable objects
        measure_obj = [getattr(self, m) for m in self.__meas_obj]
        iterable_obj = [i for i in measure_obj
                        if isinstance(i, self.__iter_types)]
        if any(iterable_obj):
            lengths = [len(i) for i in iterable_obj]
            err = "All iterable arguments must be of same length"
            if not len(set(lengths)) == 1:
                raise ValidationError(err)
            else:
                expected_len = lengths[0]
        else:
            pass

        # Dictionaries will be assumed to have the same keys after validation
        expected_keys = [f'model_{n+1}' for n in range(0, expected_len)]
        dict_obj = [i for i in iterable_obj if is_dictlike(i)]
        if any(dict_obj):
            err = "All dict arguments must have the same keys"
            if not all([k.keys() == dict_obj[0].keys() for k in dict_obj]):
                raise ValidationError(err)
            else:
                expected_keys = list(dict_obj[0].keys())
        else:
            if any(iterable_obj):
                print("Since no model names were passed, the following names",
                      "have been assigned to the models per their indexes:",
                      expected_keys)

        # All measure-related attributes will be assumed as dicts henceforth
        for name in self.__meas_obj:
            if not is_dictlike(getattr(self, name)):
                if not isinstance(getattr(self, name), self.__iter_types):
                    objL = [getattr(self, name)] * expected_len
                else:
                    objL = getattr(self, name)
                objD = {k: objL[i] for i, k in enumerate(expected_keys)}
                setattr(self, name, objD)
        return None

    def __set_predictions(self):
        """ If models are presnt, generates predictions for each model.
            Assumes that models and data have been validated.

        """
        model_objs = [*self.models.values()]
        if any(m is None for m in model_objs):
            return None
        #
        for mdl_name, mdl in self.models.items():
            pred_func = getattr(mdl, "predict", None)
            if not callable(pred_func):
                msg = f"{mdl} model does not have predict function"
                raise ValidationError(msg)
            try:
                y_pred = mdl.predict(self.X[mdl_name])
            except BaseException as e:
                msg = (f"Failure generating predictions for {mdl_name} model."
                       " Verify if data are correctly formatted for this model."
                       ) + e
                raise ValidationError(msg)
            # Since most fairness measures do not require probabilities, y_prob is
            #   optional
            try:
                y_prob = mdl.predict_proba(self.X[mdl_name])[:, 1]
            except BaseException as e:
                msg = (f"Failure predicting probabilities for {mdl_name}."
                       f" Related metrics will be skipped. {e}\n")
                warnings.warn(msg)
                y_prob = None
            #
            self.preds[mdl_name] = y_pred
            self.preds[mdl_name] = y_prob
        return None

    def __toggle_validation(self):
        self.__pause_validation = not self.__pause_validation

    def __validate(self):
        """ Verifies that attributes are set appropriately and updates as
                appropriate

        Raises:
            ValidationError
        """
        # Validation may be paused if iterated
        if self.__paused_validation():
            return None
        #
        self.__set_dicts()
        self.__validate_models()
        self.__validate_data()
        self.__set_predictions()
        return None

    def __validate_data(self):
        """ Verifies that data are of correct type and length; stores objects
            in dicts if not already in them. Assumes that models have been
            validated.

        Raises:
            ValidationError
        """
        # Ensure data dicts contain  containing pandas-compatible arrays of
        # same length
        data_obj = {m: getattr(self, m)
                    for m in self.__meas_obj if m != "models"}
        data_lengths = []
        for data_name, data in data_obj.items():
            for d in data.values():
                if data_name in ['preds', 'probs'] and d is None:
                    continue
                elif not isinstance(d, self.__data_types):
                    err = ("Input data must be numpy array or pandas object,"
                           " or a list/dict of such objects")
                    raise ValidationError(err)
                else:
                    data_lengths.append(d.shape[0])
        if not len(set(data_lengths)) == 1:
            err = "All data arguments must be of same length."
            raise ValidationError(err)

        # This class cannot yet handle grouped protected attributes or
        # multi-objective predictions
        oneD_arrays = ['prtc_attr', 'y', 'preds', 'probs']
        for name in oneD_arrays:
            data_dict = getattr(self, name)
            for _, arr in data_dict.items():
                err = None
                if name in ['preds', 'probs'] and arr is None:
                    continue
                if len(arr.shape) > 1 and arr.shape[1] > 1:
                    err = ("This library is not yet compatible with groups of"
                        " protected attributes.")

        # Measuring functions cannot yet handle continuous protected attributes
        # or targets
        binVal_arrays = ['prtc_attr', 'y', 'preds']
        for name in binVal_arrays:
            data_dict = getattr(self, name)
            for _, arr in data_dict.items():
                if name == "preds" and arr is None:
                        continue
                elif not all(np.unique(arr) == [0, 1]) and name == "prtc_attr":
                    err = (f"Expected values of [0,1] in {name}." +
                            f" Received {np.unique(arr)}")
                elif (not all(v in [0, 1] for v in np.unique(arr))
                    and name != "prtc_attr"):
                    err = (f"Expected values of [0,1] in {name}." +
                            f" Received {np.unique(arr)}")
                elif len(np.unique(arr)) > 2:
                    err = (f"Multiple labels found in {name}"
                            "(only 2 allowed).")
                if err is not None:
                    raise ValidationError(err)
        return None

    def __validate_models(self):
        """ If models are present, verifies that models have a predict function
            and generates predictions using the models. Assumes that models
            are in dict format

            Note: FairCompare should force the user to pass either models
            or predictions as None.

        Raises:
            ValidationError
        """
        # Set models as dict if not already
        if not is_dictlike(self.models):
            self.__set_dicts()

        # No validation is necessaryIf there are no models (and only
        # predictions)
        model_objs = [*self.models.values()]
        if any(m is None for m in model_objs):
            pred_objs = [*self.preds.values()]
            if any(p is None for p in pred_objs):
                if len(model_objs) == 1:
                    err = "No models or predictions defined"
                else:
                    err = ("Ether models or predictions must have a full set of"
                           + "nonmissing objects")
                raise ValidationError(err)
            # If all predictions are defined and some models are present, either
            # there is a bug or the user has changed this object's models. This
            # is only an issue if some models are missing, since their presence
            # means that all predictions will be replaced.
            elif not all(m is None for m in model_objs):
                err = "Some models are missing"
                raise ValidationError(err)
            else:
                return None
        else:
            pass
        return None


class ValidationError(Exception):
    pass
