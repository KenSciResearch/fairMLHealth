# -*- coding: utf-8 -*-
"""
Tools for measuring and comparing fairness across models

Contributors:
    camagallan <ca.magallen@gmail.com>
"""
# Copyright (c) KenSci and contributors.
# Licensed under the MIT License.

from abc import ABC
from typing import Callable, Dict, Tuple
from numbers import Number
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metric
import warnings

from .measure import summary, flag, __regression_performance
from . import __preprocessing as prep, __validation as valid
from .__validation import ArrayLike, IterableStrings, MatrixLike


def classification_performance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    target_labels: IterableStrings = None,
    sig_fig: int = 4,
):
    """ Returns a pandas dataframe of the scikit-learn classification report,
        formatted for use in fairMLHealth tools

    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
        target_labels (list of str): Optional labels for target values.
    """
    if target_labels is None:
        target_labels = [f"target = {t}" for t in set(y_true)]
    # scikit will run validation
    report = sk_metric.classification_report(
        y_true, y_pred, output_dict=True, target_names=target_labels
    )
    report = pd.DataFrame(report).transpose()
    # Move accuracy to separate row
    accuracy = report.loc["accuracy", :]
    if len(accuracy) > 0:
        report.drop("accuracy", inplace=True)
        report.loc["accuracy", "accuracy"] = accuracy[0]
    #
    report = report.round(sig_fig)
    return report


def compare(
    test_data: MatrixLike,
    targets: ArrayLike,
    protected_attr: ArrayLike,
    models: Dict[str, Callable] = None,
    predictions: ArrayLike = None,
    probabilities: ArrayLike = None,
    pred_type: str = "classification",
    flag_oor: bool = True,
    skip_performance: bool = False,
    custom_boundaries: Dict[str, Tuple[Number, Number]] = None,
    output_type: str = None,
):
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
        flag_oor (bool): if True, will apply flagging function to highlight
            fairness metrics which are considered to be outside the "fair" range
            (Out Of Range). Defaults to False.
        skip_performance (bool): If true, removes performance measures from the
            output. Defaults to False.
        custom_boundaries (dictionary{str:tuple}, optional): custom boundaries to be
            used by the flag function if requested. Keys should be measure names
            (case-insensitive).
        output_type (str): One of ["styler", "dataframe", "html", None]. Updates
            the output type of the comparison table, defaults to None, which
            returns either a pandas Dataframe (if flag_oor=False) or a pandas
            Styler (if flag_oor=True). Flagged comparisons cannot be returned as
            pandas DataFrame.

    Returns:
        pandas.Styler | pandas.DataFrame | HTML
        type determined by output_type and flag_oor arguments
    """
    comp = FairCompare(
        test_data,
        targets,
        protected_attr,
        models,
        predictions,
        probabilities,
        pred_type,
        verboseMode=True,
    )
    table = comp.compare_measures(
        flag_oor=flag_oor,
        skip_performance=skip_performance,
        output_type=output_type,
        custom_boundaries=custom_boundaries,
    )
    return table


def regression_performance(y_true: ArrayLike, y_pred: ArrayLike, sig_fig: int = 4):
    """ Returns a pandas dataframe of the regression performance metrics,
        similar to scikit's classification_performance

    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
    """
    valid.validate_array(y_true, "y_true", expected_len=None)
    valid.validate_array(y_pred, "y_pred", expected_len=len(y_true))
    y_true = prep.prep_targets(y_true)
    y_pred = prep.prep_targets(y_pred)
    if y_true.columns[0] == y_pred.columns[0]:
        y_pred.columns = ["Prediction"]
    #
    rprt_input = pd.concat([y_true, y_pred], axis=1)
    _y, _yh = rprt_input.columns[0], rprt_input.columns[1]
    report = __regression_performance(rprt_input, _y, _yh)
    report = (
        pd.DataFrame().from_dict(report, orient="index").rename(columns={0: "Score"})
    )
    report = report.round(sig_fig)
    return report


class FairCompare(ABC):
    """ Validates and stores data and models for fairness comparison
    """

    def __init__(
        self,
        test_data: MatrixLike,
        target_data: ArrayLike,
        protected_attr: ArrayLike = None,
        models: Dict = None,
        preds: ArrayLike = None,
        probs: ArrayLike = None,
        pred_type: str = "classification",
        priv_grp: int = 1,
        **kwargs,
    ):
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
        self.sig_fig = 4

        # Tool does not yet handle multiclass, but will be able to distinguish
        #   binary from multiclass in the future
        valid_pred_types = ["classification", "regression"]
        if pred_type not in valid_pred_types:
            msg = f"pred_type must be one of {valid_pred_types}. Got {pred_type}"
            raise valid.ValidationError(msg)
        self.pred_type = pred_type

        # The user is forced to pass either models or predictions as None to
        # simplify attribute management. If models are passed, they will be used
        # to produce predictions.
        self.models = models if models not in [None, [None]] else None
        self.preds = None if self.models is not None else preds
        self.probs = None if self.models is not None else probs

        #
        self.__meas_obj = [
            "X",
            "y",
            "prtc_attr",
            "priv_grp",
            "models",
            "preds",
            "probs",
        ]
        #
        if "verboseMode" in kwargs:
            self.verboseMode = kwargs.get("verboseMode")
        else:
            self.verboseMode = True
        #
        self.__setup()

    def compare_measures(
        self,
        flag_oor: bool = True,
        skip_performance: bool = False,
        custom_boundaries: Dict[str, Tuple[Number, Number]] = None,
        output_type: str = None,
    ):
        """ Returns a pandas dataframe containing fairness and performance
            measures for all available models

        Args:
            flag_oor (bool): if True, will apply flagging function to highlight
                fairness metrics which are considered to be outside the "fair"
                range
            skip_performance (bool): If true, removes performance measures from the
                output. Defaults to False.
            custom_boundaries (dictionary{str:tuple}, optional): custom boundaries to be
                used by the flag function if requested. Keys should be measure names
                (case-insensitive).
            output_type (str): One of ["styler", "dataframe", "html", None].
                Updates the output type of the comparison table, defaults to None,
                which returns either a pandas Dataframe (if flag_oor=False) or a
                pandas Styler (if flag_oor=True). Flagged comparisons cannot be
                returned as pandas DataFrame.

        Returns:
            pandas.Styler | pandas.DataFrame | HTML
            type determined by output_type and flag_oor arguments
        """
        # Model objects are assumed to be held in a dict
        if not valid.is_dictlike(self.models):
            self.__set_dicts()
        #
        self.__validate_output_type(output_type, flag_oor)
        if output_type is None and flag_oor is False:
            output_type = "dataframe"
        output_type = "" if output_type is None else output_type.lower()
        #
        if len(self.models) == 0:
            warnings.warn("No models to compare.")
            return pd.DataFrame()
        else:
            test_results = []
            self.__toggle_validation()
            # Compile measure_model results for each model
            for model_name in self.models.keys():
                # Keep flag off at this stage to allow column rename (flagger
                # returns a pandas Styler). Flag applied a few lines below
                res = self.measure_model(
                    model_name,
                    skip_performance=skip_performance,
                    flag_oor=False,
                    custom_boundaries=custom_boundaries,
                )
                res.rename(columns={"Value": model_name}, inplace=True)
                test_results.append(res)
            self.__toggle_validation()  # toggle-on model validation
            if len(test_results) > 0:
                output = pd.concat(test_results, axis=1)
                if flag_oor:
                    as_styler = True if output_type != "html" else False
                    output = flag(output, sig_fig=4, as_styler=as_styler)
                else:
                    if output_type.lower() == "styler":
                        output = output.style
                    elif output_type.lower() == "html":
                        output = output.to_html()
                    else:
                        pass
            else:
                output = None
            return output

    def measure_model(
        self,
        model_name: str,
        custom_boundaries: Dict[str, Tuple[Number, Number]] = None,
        **kwargs,
    ):
        """ Creates a table of fairness-related measures for the model_name specified

        Args:
            model_name (str): a key corresponding to the model of interest,
                as found in the object's "models" dictionary
            custom_boundaries (dictionary{str:tuple}, optional): custom boundaries to be
                used by the flag function if requested. Keys should be measure names
                (case-insensitive).

        Returns:
            pandas.DataFrame
        """
        self.__validate(model_name)
        msg = f"Could not measure fairness for {model_name}"
        if model_name not in self.preds.keys():
            msg += (
                " Name not found Available options include "
                f"{list(self.preds.keys())}"
            )
            print(msg)
            return pd.DataFrame()
        elif self.preds[model_name] is None:
            msg += " No predictions present."
            print(msg)
            return pd.DataFrame()
        else:
            res = summary(
                X=self.X[model_name],
                y_true=self.y[model_name],
                y_pred=self.preds[model_name],
                y_prob=self.probs[model_name],
                prtc_attr=self.prtc_attr[model_name],
                pred_type=self.pred_type,
                sig_fig=self.sig_fig,
                custom_ranges=custom_boundaries,
                **kwargs,
            )
            return res

    def __validate_output_type(
        self, output_type: str = None, flag_request: bool = None
    ):
        valid_outputs = ["styler", "html", "dataframe", None]
        if not isinstance(output_type, str) and output_type is not None:
            raise TypeError(f"output_type must be string, one of {valid_outputs}")
        elif output_type is None and flag_request is False:  # acceptable combination
            return None
        # test output_type as string to facilitate remaining
        output_str = "" if output_type is None else output_type.lower()
        if output_str not in valid_outputs and output_type is not None:
            raise ValueError(f"output_type must be one of {valid_outputs}")
        elif output_str == "dataframe" and flag_request == True:
            msg = "Flags can only be used for html or pandas styler outputs, not dataframe"
            raise ValueError(msg)
        else:
            return None

    def __check_models_predictions(self, enforce: bool = True):
        """ If any predictions are missing, generates predictions for each model.
            Assumes that models and data have been validated.

        """
        # Model objects are assumed to be held in a dict
        if not valid.is_dictlike(self.models):
            self.__set_dicts()
        #
        model_objs = [*self.models.values()]
        pred_objs = [*self.preds.values()]
        prob_objs = [*self.probs.values()]
        missing_probs = []
        has_probs = None
        #
        if enforce:
            if not any(m is None for m in model_objs):
                for mdl_name, mdl in self.models.items():
                    pred_func = getattr(mdl, "predict", None)
                    if not callable(pred_func):
                        msg = f"{mdl} model does not have predict function"
                        raise valid.ValidationError(msg)
                    try:
                        y_pred = mdl.predict(self.X[mdl_name])
                    except BaseException as e:
                        e = getattr(e, "message") if "message" in dir(e) else str(e)
                        msg = (
                            f"Failure generating predictions for {mdl_name}"
                            " model. Verify if data are correctly formatted"
                            " for this model."
                        ) + e
                        raise valid.ValidationError(msg)
                    self.preds[mdl_name] = y_pred
                    # Since most fairness measures do not require probabilities,
                    #   y_prob is optional
                    has_probs = getattr(mdl, "predict_proba", None)
                    if has_probs is not None:
                        try:
                            y_prob = mdl.predict_proba(self.X[mdl_name])[:, 1]
                        except BaseException:
                            y_prob = None
                            missing_probs.append(mdl_name)
                        self.probs[mdl_name] = y_prob
                    else:
                        pass
            elif not all(m is None for m in model_objs):
                raise valid.ValidationError(
                    "Incomplete set of models detected. Can't process a mix of"
                    + " models and predictions"
                )
            else:
                if self.pred_type == "regression":
                    pass
                elif any(p is None for p in pred_objs):
                    raise valid.ValidationError(
                        "Cannot measure without either models or predictions"
                    )
                elif has_probs is not None:
                    missing_probs = [p for p in prob_objs if p is None]
        else:
            if self.pred_type == "regression":
                pass
            elif any(p is None for p in pred_objs):
                raise valid.ValidationError(
                    "Cannot measure without either models or predictions"
                )
            elif has_probs is not None:
                missing_probs = [p for p in prob_objs if p is None]

        if self.pred_type == "classification" and any(missing_probs):
            warnings.warn(
                "Please note that probabilities could not be "
                + f"generated for the following models: {missing_probs}. "
                + "Dependent metrics will be skipped."
            )

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
        expected_len = 1  # expected len of iterable objects
        expected_keys = []

        # Iterable attributes must be of same length so that keys can be
        # properly matched when they're converted to dictionaries.
        iterable_obj = [
            m for m in self.__meas_obj if isinstance(getattr(self, m), valid.ITER_TYPES)
        ]
        if any(iterable_obj):
            lengths = [len(getattr(self, i)) for i in iterable_obj]
            err = "All iterable arguments must be of same length"
            if not len(set(lengths)) == 1:
                raise valid.ValidationError(err)
            else:
                expected_len = lengths[0]

        # Dictionaries will assume the same keys after validation
        dict_obj = [
            getattr(self, i)
            for i in iterable_obj
            if valid.is_dictlike(getattr(self, i))
        ]
        if any(dict_obj):
            err = "All dict arguments must have the same keys"
            if not all([k.keys() == dict_obj[0].keys() for k in dict_obj]):
                raise valid.ValidationError(err)
            elif not any(expected_keys):
                expected_keys = list(dict_obj[0].keys())
        else:
            expected_keys = [f"model {n+1}" for n in range(0, expected_len)]

        # All measure-related attributes will be assumed as dicts henceforth
        for name in self.__meas_obj:
            if not valid.is_dictlike(getattr(self, name)):
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
        """ Validates models and data necessary to generate predictions. Then,
            generates predictions using those models as needed. To be run on
            initialization only, or whenever model objects are updated, so that
            predictions are not updated
        """
        try:
            if not (self.models is None or self.preds is None):
                err = (
                    "FairCompare accepts either models or predictions, but" + "not both"
                )
                raise valid.ValidationError(err)
            self.__set_dicts()
            for x in self.X.values():
                valid.validate_analytical_input(x)
            self.__check_models_predictions()
            for m in self.models.keys():
                self.__validate(m)
        except valid.ValidationError as ve:
            raise valid.ValidationError(f"Error loading FairCompare. {ve}")

    def __toggle_validation(self):
        self.__pause_validation = not self.__pause_validation

    def __validate(self, model_name: str):
        """ Verifies that attributes are set appropriately and updates as
                appropriate

        Raises:
            valid.ValidationError
        """
        # Validation may be paused during iteration to save time
        if self.__paused_validation():
            return None
        else:
            self.__check_models_predictions(enforce=False)
            valid.validate_analytical_input(
                X=self.X[model_name],
                y_true=self.y[model_name],
                y_pred=self.preds[model_name],
                y_prob=self.probs[model_name],
                prtc_attr=self.prtc_attr[model_name],
                priv_grp=self.priv_grp[model_name],
            )
            return None
