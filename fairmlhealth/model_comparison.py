# -*- coding: utf-8 -*-
"""
Tools for measuring and comparing fairness across models

Contributors:
    camagallan <ca.magallen@gmail.com>
"""
# Copyright (c) KenSci and contributors.
# Licensed under the MIT License.

from abc import ABC
import pandas as pd
import warnings

from .utils import is_dictlike
from .reports import summary_report, flag
from . import __validation as valid
from .__validation import ValidationError


"""
    Model Comparison Tools
"""


def measure_model(test_data, targets, protected_attr, model=None,
                  predictions=None, probabilities=None,
                  pred_type="classification", flag_oor=True):
    """ Generates a report of fairness measures for the model

    Args:
        test_data (pandas DataFrame or compatible type):
        targets (1D array-like):
        protected_attr (1D array-like):
        model (scikit model or other model object with a *.predict() function
            Defaults to None. If None, must pass predictions.
        predictions (1D array-like): Set of predictions
            corresponding to targets. Defaults to None. Ignored
            if model argument is passed.
        probabilities (1D array-like): Set of probabilities
            corresponding to predictions. Defaults to None. Ignored
            if models argument is passed.
        flag_oor (bool): if true, will apply flagging function to highlight
            fairness metrics which are considered to be outside the "fair" range
            (Out Of Range). Defaults to False.

    Returns:
        pandas dataframe of fairness measures for the model
    """
    comp = FairCompare(test_data, targets, protected_attr, model,
                       predictions, probabilities, pred_type, verboseMode=True)
    model_name = list(comp.models.keys())[0]
    table = comp.measure_model(model_name, flag_oor=flag_oor,
                               skip_performance=True)
    return table


def compare_models(test_data, targets, protected_attr, models=None,
                   predictions=None, probabilities=None,
                   pred_type="classification", flag_oor=True):
    """ Generates a report comparing fairness measures for the models passed.
            Note: This is a wrapper for the FairCompare.compare_measures method
            See FairCompare for more information.

    Args:
        test_data (pandas DataFrame or compatible type):
        targets (1D array-like):
        protected_attr (1D array-like):
        model (scikit models or other model objects with a *.predict() function
            that accept test_data and return an array of predictions).
            Defaults to None. If None, must pass predictions.
        predictions (1D array-likes): Set of predictions
            corresponding to targets. Defaults to None. Ignored
            if model argument is passed.
        probabilities (1D array-like): Set of probabilities
            corresponding to predictions. Defaults to None. Ignored
            if models argument is passed.
        flag_oor (bool): if true, will apply flagging function to highlight
            fairness metrics which are considered to be outside the "fair" range
            (Out Of Range). Defaults to False.

    Returns:
        pandas dataframe of fairness and performance measures for each model
    """
    comp = FairCompare(test_data, targets, protected_attr, models,
                       predictions, probabilities, pred_type, verboseMode=True)
    table = comp.compare_measures(flag_oor=flag_oor)
    return table

class FairCompare(ABC):
    """ Validates and stores data and models for fairness comparison
    """

    def __init__(self, test_data, target_data, protected_attr=None,
                 models=None, preds=None, probs=None,
                 pred_type="classification", priv_grp=1,  **kwargs):
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
            preds (1D array-like): Set of predictions
                corresponding to targets. Defaults to None. Ignored
                if model argument is passed.
            probs (1D array-like): Set of probabilities
                corresponding to predictions. Defaults to None. Ignored
                if models argument is passed.
        """
        #
        self.X = test_data
        self.prtc_attr = protected_attr
        self.priv_grp = priv_grp
        self.y = target_data
        self.pred_type = pred_type
        self.sig_fig = 4

        # The user is forced to pass either models or predictions as None to
        # simplify attribute management. If models are passed, they will be used
        # to produce predictions.
        self.models = models if models not in [None, [None]] else None
        self.preds = None if self.models is not None else preds
        self.probs = None if self.models is not None else probs

        #
        self.__meas_obj = ["X", "y", "prtc_attr", "priv_grp", "models",
                           "preds", "probs"]
        #
        if "verboseMode" in kwargs:
            self.verboseMode = kwargs.get("verboseMode")
        else:
            self.verboseMode = True
        #
        self.__setup()

    def compare_measures(self, flag_oor=False):
        """ Returns a pandas dataframe containing fairness and performance
            measures for all available models

        Args:
            flag_oor (bool): if true, will apply flagging function to highlight
            fairness metrics which are considered to be outside the "fair" range
            (Out Of Range)
        """
        # Model objects are assumed to be held in a dict
        if not is_dictlike(self.models):
            self.__set_dicts()
        #
        if len(self.models) == 0:
            warnings.warn("No models to compare.")
            return pd.DataFrame()
        else:
            test_results = []
            self.__toggle_validation()
            # Compile measure_model results for each model
            for model_name in self.models.keys():
                res = self.measure_model(model_name, flag_oor=False)
                res.rename(columns={'Value': model_name}, inplace=True)
                test_results.append(res)
            self.__toggle_validation()  # toggle-on model validation
            if len(test_results) > 0:
                output = pd.concat(test_results, axis=1)
                if flag_oor:
                    output = flag(output, sig_fig=4)
            else:
                output = None
            return output

    def measure_model(self, model_name, **kwargs):
        """ Returns a pandas dataframe containing fairness measures for the
                model_name specified

        Args:
            model_name (str): a key corresponding to the model of interest,
                as found in the object's "models" dictionary
        """
        self.__validate(model_name)
        msg = f"Could not measure fairness for {model_name}"
        if model_name not in self.preds.keys():
            msg += (" Name not found Available options include "
                   f"{list(self.preds.keys())}")
            print(msg)
            return pd.DataFrame()
        elif self.preds[model_name] is None:
            msg += (" No predictions present.")
            print(msg)
            return pd.DataFrame()
        else:
            res = summary_report(self.X[model_name],
                                 self.prtc_attr[model_name],
                                 self.y[model_name],
                                 self.preds[model_name],
                                 self.probs[model_name],
                                 pred_type=self.pred_type,
                                 sig_fig=self.sig_fig,
                                 **kwargs)
            return res

    def __check_models_predictions(self, enforce=True):
        """ If any predictions are missing, generates predictions for each model.
            Assumes that models and data have been validated.

        """
        # Model objects are assumed to be held in a dict
        if not is_dictlike(self.models):
            self.__set_dicts()
        #
        model_objs = [*self.models.values()]
        pred_objs = [*self.preds.values()]
        prob_objs = [*self.probs.values()]
        missing_probs = []
        #
        if enforce:
            if not any(m is None for m in model_objs):
                for mdl_name, mdl in self.models.items():
                    pred_func = getattr(mdl, "predict", None)
                    if not callable(pred_func):
                        msg = f"{mdl} model does not have predict function"
                        raise ValidationError(msg)
                    try:
                        y_pred = mdl.predict(self.X[mdl_name])
                    except BaseException as e:
                        e = getattr(e, 'message') if 'message' in dir(e) else str(e)
                        msg = (f"Failure generating predictions for {mdl_name}"
                               " model. Verify if data are correctly formatted"
                               " for this model.") + e
                        raise ValidationError(msg)
                    self.preds[mdl_name] = y_pred
                    # Since most fairness measures do not require probabilities,
                    #   y_prob is optional
                    try:
                        y_prob = mdl.predict_proba(self.X[mdl_name])[:, 1]
                    except BaseException:
                        y_prob = None
                        missing_probs.append(mdl_name)
                    self.probs[mdl_name] = y_prob

            elif not all(m is None for m in model_objs):
                raise ValidationError(
                    "Incomplete set of models detected. Can't process a mix of"
                    + " models and predictions")
            else:
                if any(p is None for p in pred_objs):
                    raise ValidationError(
                        "Cannot measure without either models or predictions")
                missing_probs = [p for p in prob_objs if p is None]
        else:
            if any(p is None for p in pred_objs):
                raise ValidationError(
                        "Cannot measure without either models or predictions")
            missing_probs = [p for p in prob_objs if p is None]

        if any(missing_probs):
            warnings.warn("Please note that probabilities could not be " +
                f"generated for the following models: {missing_probs}. " +
                "Dependent metrics will be skipped.")

        return None

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
        # Until otherwise updated, expect all objects to be non-iterable and
        # assume no keys
        expected_len = 1 # expected len of iterable objects
        expected_keys = []

        # Iterable attributes must be of same length so that keys can be
        # properly matched when they're converted to dictionaries.
        iterable_obj = [m for m in self.__meas_obj
                        if isinstance(getattr(self, m), valid.ITER_TYPES)]
        if any(iterable_obj):
            lengths = [len(getattr(self, i)) for i in iterable_obj]
            err = "All iterable arguments must be of same length"
            if not len(set(lengths)) == 1:
                raise ValidationError(err)
            else:
                expected_len = lengths[0]

        # Dictionaries will assume the same keys after validation
        dict_obj = [getattr(self, i)
                    for i in iterable_obj if is_dictlike(getattr(self, i))]
        if any(dict_obj):
            err = "All dict arguments must have the same keys"
            if not all([k.keys() == dict_obj[0].keys() for k in dict_obj]):
                raise ValidationError(err)
            elif not any(expected_keys):
                expected_keys = list(dict_obj[0].keys())
        else:
            expected_keys = [f'model {n+1}' for n in range(0, expected_len)]

        # All measure-related attributes will be assumed as dicts henceforth
        for name in self.__meas_obj:
            if not is_dictlike(getattr(self, name)):
                if not isinstance(getattr(self, name), valid.ITER_TYPES):
                    objL = [getattr(self, name)] * expected_len
                else:
                    objL = getattr(self, name)
                objD = {k: objL[i] for i, k in enumerate(expected_keys)}
                setattr(self, name, objD)
            else:
                pass
        return None

    def __setup(self):
        ''' Validates models and data necessary to generate predictions. Then,
            generates predictions using those models as needed. To be run on
            initialization only, or whenever model objects are updated, so that
            predictions are not updated
        '''
        try:
            if not (self.models is None or self.preds is None):
                err = ("FairCompare accepts either models or predictions, but" +
                       "not both")
                raise ValidationError(err)
            self.__set_dicts()
            for x in self.X.values():
                valid.validate_report_input(x)
            self.__check_models_predictions()
            for m in self.models.keys():
                self.__validate(m)
        except ValidationError as ve:
            raise ValidationError(f"Error loading FairCompare. {ve}")


    def __toggle_validation(self):
        self.__pause_validation = not self.__pause_validation

    def __validate(self, model_name):
        """ Verifies that attributes are set appropriately and updates as
                appropriate

        Raises:
            ValidationError
        """
        # Validation may be paused during iteration to save time
        if self.__paused_validation():
            return None
        else:
            self.__check_models_predictions(enforce=False)
            valid.validate_report_input(X=self.X[model_name],
                                        y_true=self.y[model_name],
                                        y_pred=self.preds[model_name],
                                        y_prob=self.probs[model_name],
                                        prtc_attr=self.prtc_attr[model_name],
                                        priv_grp=self.priv_grp[model_name]
                                        )
            return None
