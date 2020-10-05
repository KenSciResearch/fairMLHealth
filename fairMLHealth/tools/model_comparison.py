''' Tools for measuring and comparing fairness across models

'''
# Contributors: Christine Allen <christine.allen@kensci.com>
# Copyright (c) KenSci and contributors.
# Licensed under the MIT License.

from abc import ABC
import aif360.sklearn.metrics as aif_mtrc
import fairlearn.metrics as fl_mtrc
import fairMLHealth.tools.measures as fh_mtrc
from IPython.display import HTML
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metric
import warnings
# Temporarily hide pandas SettingWithCopy warning
warnings.filterwarnings('ignore', module='pandas' )
warnings.filterwarnings('ignore', module='sklearn' )





'''
    Model Comparison Tools
'''


def compare_models(test_data, target_data, protected_attr_data=None, models=None):
    """ Generates a report comparing fairness measures for the models passed.
            Note: This is a wrapper for the fairCompare.compare_models method.
            See fairCompare for more information.

        Returns:
            a pandas dataframe
    """
    comp = fairCompare(test_data, target_data, protected_attr_data, models)
    table = comp.compare_models()
    return(table)

class fairCompare(ABC):
    """ Validates and stores data and models for fairness comparison

        TODO: inherit AIF360 data object
    """
    def __init__(self, test_data, target_data, protected_attr_data=None,
                                                                    models=None):
        """ Validates and attaches attributes

            Args:
                test_data (numpy array or similar pandas object): data to be
                    passed to the models to generate predictions. It's
                    recommended that these be separate data from those used to
                    train the model.
                target_data (numpy array or similar pandas object): target data
                    array corresponding to the test data. It is recommended that
                    the target is not present in the test_data.
                protected_attr_data (numpy array or similar pandas object):
                    protected attributes that may or may not be present in
                    test_data. Note that values must currently be binary or
                    boolean type
                models (dict or list-like): the set of trained models to be
                    evaluated. Dict keys assumed as model names. If a list-like
                    object is passed, will set model names relative to their index
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
                TypeError: data must by scikit-compatible format
                ValueError: data must be of same length
        """
        # Skip validation if paused
        if self.__validation_paused():
            return None
        #
        valid_data_types = (pd.DataFrame, pd.Series, np.ndarray)
        for data in [self.X, self.y]:
            if not isinstance(data, valid_data_types):
                raise TypeError("input data must be numpy array" +
                                    " or similar pandas object")
        if not self.X.shape[0] == self.y.shape[0]:
            raise ValueError("test and target data mismatch")
        # Ensure that every column of the protected attributes is boolean
        if self.protected_attr is not None:
            if not isinstance(self.protected_attr,valid_data_types):
                raise TypeError("Protected attribute(s) must be numpy array" +
                                    " or similar pandas object")
        # Validate models and ensure as dict
        if not isinstance(self.models, (dict)) and self.models is not None:
            if not isinstance(self.models, (list, tuple, set)):
                raise TypeError("Models must be dict or list-like group of" +
                    " trained, scikit-compatible models")
            self.models = {f'model_{i}':m for i,m in enumerate(self.models)}
            print("Since no model names were passed, the following names have",
                  "been assigned to the models per their indexes:",
                  f"{list(self.models.keys())}")
        if self.models is not None:
            if not len(self.models) > 0:
                raise  "The set of models is empty"
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
            print(f"Error measuring fairness: {model_name} does not appear in" +
                  f" the models. Available models include {list(self.models.keys())}")
            return pd.DataFrame()
        m = self.models[model_name]
        # Cannot measure fairness without predictions
        try:
            y_pred =  m.predict(self.X)
        except:
            try:
                y_pred =  m.predict(self.X.to_numpy())
            except:
                raise ValueError(f"Error generating predictions for {model_name}" +
                    " Check that it is a trained, scikit-compatible model" +
                    " that can generate predictions using the test data")
        # Since most fairness measures do not require probabilities, y_prob is optional
        try:
            y_prob = m.predict_proba(self.X)[:,1]
        except:
            print(f"Failure predicting probabilities for {model_name}." +
                  " Related metrics will be skipped.")
            y_prob = None
        finally:
            res = report_classification_fairness(self.X, self.protected_attr,
                                                 self.y, y_pred, y_prob)
            return res

    def compare_models(self):
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
            self.__toggle_validation()
            for model_name in self.models.keys():
                res = self.measure_model(model_name)
                if res is None:
                    continue
                    print("ping")
                else:
                    res.rename(columns={'Value':model_name}, inplace=True)
                    test_results.append(res)
            self.__toggle_validation()
            if len(test_results) > 0:
                output = pd.concat(test_results, axis=1)
                return output
            else:
                return None

