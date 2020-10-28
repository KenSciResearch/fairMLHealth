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
        self.__validate()

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
        for data in [self.X, self.y]:
            if not isinstance(data, valid_data_types):
                raise ValidationError("input data must be numpy array" +
                                      " or similar pandas object")
        if not self.X.shape[0] == self.y.shape[0]:
            raise ValidationError("test and target data mismatch")
        # Ensure that every column of the protected attributes is boolean
        if self.protected_attr is not None:
            if not isinstance(self.protected_attr, valid_data_types):
                raise ValidationError("Protected attribute(s) must be numpy"
                                      + " array or similar pandas object")
            data_shape = self.protected_attr.shape
            if len(data_shape) > 1 and data_shape[1] > 1:
                raise ValidationError("This library is not yet compatible with"
                                      + " multiple protected attributes."
                                    )
        # Ensure models appear as dict
        if not isinstance(self.models, (dict)) and self.models is not None:
            if not isinstance(self.models, (list, tuple, set)):
                raise ValidationError("Models must be dict or list-like group"
                                      + " of trained, skikit-like models")
            self.models = {f'model_{i}': m for i, m in enumerate(self.models)}
            print("Since no model names were passed, the following names have",
                  "been assigned to the models per their indexes:",
                  f"{list(self.models.keys())}")
        # Ensure that models are present and have a predict function
        if self.models is not None:
            if not len(self.models) > 0:
                raise ValidationError("The set of models is empty")
            else:
                for _, m in self.models.items():
                    pred_func = getattr(m, "predict", None)
                    if not callable(pred_func):
                        raise ValidationError(
                                    f"{m} model does not have predict function")
        return None

    def __validation_paused(self):
        if not hasattr(self, "__pause_validation"):
            self.__pause_validation = False
        return self.__pause_validation

    def __toggle_validation(self):
        if self.__pause_validation:
            self.__pause_validation = False
        else:
            self.__pause_validation = True

    def measure_model(self, model_name):
        """ Generates a report comparing fairness measures for the model_name
                specified

            Returns:
                a pandas dataframe
        """
        self.__validate()
        if model_name not in self.models.keys():
            print(f"Error measuring fairness: {model_name} does not appear in",
                  "the models. Available models include ",
                  f"{list(self.models.keys())}")
            return pd.DataFrame()
        m = self.models[model_name]
        # Verify that predictions can be generated from the test data
        try:
            y_pred = m.predict(self.X)
        except BaseException as e:
            raise ValidationError("Failure generating predictions for " +
                                  f"{model_name}. Verify that data are " +
                                  f"correctly formatted for this model. {e}")
        # Since most fairness measures do not require probabilities, y_prob is
        #   optional
        try:
            y_prob = m.predict_proba(self.X)[:, 1]
        except BaseException as e:
            warnings.warn(f"Failure predicting probabilities for {model_name}."
                          + f" Related metrics will be skipped. {e}")
            y_prob = None
        finally:
            res = reports.classification_fairness(self.X, self.protected_attr,
                                                  self.y, y_pred, y_prob)
            return res

    def compare_measures(self):
        """ Generates a report comparing fairness measures for all available
                models

        Returns:
            a pandas dataframe
        """
        self.__validate()
        if len(self.models) == 0:
            print("No models to compare.")
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

    def save_comparison(self, filepath):
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        dump(self, filepath)


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
