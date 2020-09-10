from abc import ABC
from aif360.sklearn.metrics import *
import pandas as pd
import numpy as np
import sklearn.metrics as skmetric
import warnings
# Temporarily hide pandas SettingWithCopy warning
warnings.filterwarnings('ignore', module='pandas' )



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
            print("No model names passed. The following names have been assigned" +
                  f" to the models according to their indexes: {list(self.models.keys())}")
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
                    res.set_index('Measure', inplace=True)
                    res.rename(columns={'Value':model_name}, inplace=True)
                    test_results.append(res)
            self.__toggle_validation()
            if len(test_results) > 0:
                output = pd.concat(test_results, axis=1)
                return output
            else:
                return None



def report_classification_fairness(X, protected_attr, y_true, y_pred, y_prob=None):
    """ Returns a dataframe containing fairness measures for the model results

        Args:
            X (array-like): Sample features
            protected_attr (array-like, named): values for the protected attribute
                (note: protected attribute may also be present in X)
            y_true (array-like, 1-D): Sample targets
            y_pred (array-like, 1-D): Sample target probabilities
            protected_attr (list): list of column names or locations in
                X containing the protected attribute(s) against which
                fairness is measured

        Returns:
            pandas dataframe
    """
    valid_data_types = (pd.DataFrame, pd.Series, np.ndarray)
    for data in [X, protected_attr, y_true, y_pred]:
        if not isinstance(data, valid_data_types):
            raise TypeError("input data is invalid type")
        if not data.shape[0] > 1:
            raise ValueError("input data is too small to measure")
    if y_prob is not None:
        if not isinstance(y_prob, valid_data_types):
            raise TypeError("y_prob is invalid type")

    # Format inputs to required datatypes
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(protected_attr, (np.ndarray, pd.Series)):
        if isinstance(protected_attr, pd.Series):
            protected_attr = pd.DataFrame(protected_attr,
                                          columns=[protected_attr.name])
        else:
            protected_attr = pd.DataFrame(protected_attr)
    if isinstance(y_true, (np.ndarray, pd.Series)):
        y_true = pd.DataFrame(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred)
    if isinstance(y_prob, np.ndarray):
        y_prob = pd.DataFrame(y_prob)
    for data in [y_true, y_pred, y_prob]:
        if data.shape[1] > 1:
            raise TypeError("targets and predictions must be 1-Dimensional")

    # Ensure that protected_attr is integer-valued
    pa_cols = protected_attr.columns.tolist()
    for c in pa_cols:
        binary = ( set(protected_attr[c].astype(int)) == set(protected_attr[c]) )
        boolean = ( protected_attr[c].dtype == bool )
        two_valued = ( set(protected_attr[c].astype(int)) == {0,1} )
        if not two_valued and (binary or boolean):
            raise ValueError(
            "protected_attr must be binary or boolean and heterogeneous")
        protected_attr.loc[:, c] = protected_attr[c].astype(int)
        if isinstance(c, int):
            protected_attr.rename(columns={c:f"protected_attribute_{c}"}, inplace=True)

    # Format and set sensitive attributes as index for y dataframes
    pa_name = protected_attr.columns.tolist()
    protected_attr.reset_index(inplace=True, drop=True)
    y_true = pd.concat([protected_attr, y_true.reset_index(drop=True)], axis=1).set_index(pa_name)
    y_pred = pd.concat([protected_attr, y_pred.reset_index(drop=True)], axis=1).set_index(pa_name)
    y_prob = pd.concat([protected_attr, y_prob.reset_index(drop=True)], axis=1).set_index(pa_name)
    y_pred.columns = y_true.columns
    y_prob.columns = y_true.columns

    # Generate lists of performance measures to be converted to dataframe
    scores = []

    #
    scores.append( ['** Group Fairness **', None])
    scores.append( ['Statistical Parity Difference',
                        statistical_parity_difference(y_true, y_pred,
                                                      prot_attr=pa_name)] )
    scores.append( ['Disparate Impact Ratio',
                        disparate_impact_ratio(y_true, y_pred,
                                    prot_attr=pa_name)] )
    scores.append( ['Average Odds Difference',
                        average_odds_difference(y_true, y_pred,
                                    prot_attr=pa_name)] )
    scores.append( ['Equal Opportunity Difference',
                        equal_opportunity_difference(y_true, y_pred,
                                    prot_attr=pa_name)] )
    if y_prob is not None:
        scores.append( ['Positive Predictive Parity Difference',
                        difference(skmetric.precision_score, y_true, y_pred,
                                     prot_attr=pa_name, priv_group=1)] )
        scores.append( ['Between-Group AUC Difference',
                        difference(skmetric.roc_auc_score, y_true, y_prob,
                                   prot_attr=pa_name, priv_group=1)] )
        scores.append( ['Between-Group Balanced Accuracy Difference',
                        difference(skmetric.balanced_accuracy_score, y_true,
                                   y_pred, prot_attr=pa_name, priv_group=1)] )
    else:
        pass
    #
    scores.append( ['** Individual Fairness **', None])
    scores.append( ['Consistency Score', consistency_score(X, y_pred.iloc[:,0])] )
    scores.append( ['Between-Group Generalized Entropy Error',
                        between_group_generalized_entropy_error(y_true, y_pred,
                                                                prot_attr=pa_name)])

    #
    scores.append( ['** Model Performance **', None])
    target_labels = [f"target = {t}" for t in set(np.unique(y_true))]
    report = report_scikit(y_true.iloc[:,0], y_pred.iloc[:,0], target_labels)
    avg_lbl = "avg / total" if len(target_labels) > 2 else target_labels[-1]
    scores.append( ['Precision', report.loc[avg_lbl, 'precision']])
    scores.append( ['Recall', report.loc[avg_lbl, 'recall']])
    scores.append( ['F1-Score', report.loc[avg_lbl, 'f1-score']])
    if len(target_labels) == 2:
        scores.append( ['Accuracy', report.loc[avg_lbl, 'accuracy']])

    # Convert scores to a formatted dataframe and return
    model_scores =  pd.DataFrame(scores, columns=['Measure','Value'])
    model_scores['Value'] = model_scores.loc[:,'Value'].round(4)
    model_scores.fillna("", inplace=True)
    return model_scores



def report_scikit(y_true, y_pred, target_labels=None):
    """ Returns a formatted dataframe of the scikit-learn classification report

    Args:
        y_true (scikit-compatible array): target values
        y_pred (scikit-compatible array): prediction values
        target_labels (list of str): optional labels to elucidate target values
    """
    if target_labels is None:
        target_labels = [f"target = {t}" for t in set(y_true)]
    report = skmetric.classification_report(y_true, y_pred, output_dict=True,
                                            target_names=target_labels)
    report = pd.DataFrame(report).transpose()
    if len(target_labels) == 2:
        accuracy = report.loc['accuracy',:]
        report.drop('accuracy', inplace=True)
        report.loc[target_labels[1], 'accuracy'] = accuracy[0]
    return(report)

