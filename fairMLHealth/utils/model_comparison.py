from abc import ABC
from aif360.sklearn.metrics import *
import pandas as pd
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score, precision_score)



def compare_models(test_data, target_data, protected_attr_data=None, models=[None]):
    """[summary]

    Args:
        test_data ([type]): [description]
        target_data ([type]): [description]
        protected_attr_data ([type], optional): [description]. Defaults to None.
        models (list, optional): [description]. Defaults to [None].

    Returns:
        pandas dataframe: [description]
    """
    comp = fairCompare(test_data, target_data, protected_attr_data, models)
    table = comp.compare_models()
    return(table)


class fairCompare(ABC):
    """ Validates and stores data and models for fairness comparison

        TODO: inherit AIF360 data object
    """
    def __init__(self, test_data, target_data, protected_attr_data=None, models=[None]):
        """ Validates and attaches attributes

            Args:
                test_data (array-like): test data
                target_data (array_like): target data array corresponding to the
                    test data - these data should not be present in the test data
                protected_attr_data (array-like): protected attributes that
                    may or may not be present in test_data
                models (list or dict-like): the set of models to be evaluated.
                    Dict keys assumed as model names. If a list-like object is
                    passed, will set each model name as the index+1
        """
        self.__validate(test_data, target_data, protected_attr_data, models)
        # Attach attributes
        self.X = test_data
        self.protected_attr = protected_attr_data
        self.y = target_data
        self.models = models


    def __validate(self, test_data, target_data, protected_attr, models):
        if not isinstance(models, dict):
            assert isinstance(models, (list, tuple, set)), (
                "models must be dict or list-like group of trained,"
                "scikit-compatible models")
            models = {str(i+1):m for i,m in enumerate(models)}
        assert test_data.shape[0] == target_data.shape[0], (
            "test and target data mismatch")
        if protected_attr is not None:
            assert set(protected_attr) == {0,1}, "protected_attr must be bool"
        return None


    def measure_fairness(self, model_name):
        ''' Returns a dataframe of fairness measures for the model
        '''
        m = self.models[model_name]
        res = get_fairness_measures_df(self.X, self.protected_attr, self.y,
                                    m.predict(self.X), m.predict_proba(self.X))
        return res

    def compare_models(self):
        """ Returns a dataframe comparing fairness measures for all available 
                models
        """
        test_results = []
        for model_name in self.models.keys():
            res = self.measure_fairness(model_name)
            res.set_index('Measure', inplace=True)
            res.rename(columns={'Value':model_name}, inplace=True)
            test_results.append(res)

        output = pd.concat(test_results, axis=1)
        return output





def get_fairness_measures_df(X, protected_attr, y_true, y_pred, y_prob=None):
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
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(protected_attr, (np.ndarray, pd.Series)):
        if isinstance(protected_attr, pd.Series):
            protected_attr = pd.DataFrame(protected_attr, columns=[protected_attr.name])
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    if isinstance(y_prob, np.ndarray):
        if len(np.shape(y_prob)) == 2:
            y_prob = y_prob[:, 1]
        y_prob = pd.Series(y_prob)
    # Format and set sensitive attributes as index for y dataframes
    pa_name = protected_attr.columns.tolist()
    protected_attr.reset_index(inplace=True, drop=True)
    y_true.reset_index(inplace=True, drop=True)
    y_true = pd.concat([protected_attr, y_true], axis=1).set_index(pa_name)
    y_pred = pd.concat([protected_attr, y_pred], axis=1).set_index(pa_name)
    y_prob = pd.concat([protected_attr,y_prob], axis=1).set_index(pa_name)
    y_pred.columns = y_true.columns
    y_prob.columns = y_true.columns
    # Generate lists of performance measures to be converted to dataframe
    scores = []
    scores.append( ['** Group Measures **', None])
    scores.append( ['Statistical Parity Difference',
                        statistical_parity_difference(y_true, y_pred, prot_attr=pa_name)] )
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
                          difference(precision_score, y_true, y_pred,
                                     prot_attr=pa_name, priv_group=1)] )
        scores.append( ['Between-Group AUC Difference',
                        difference(roc_auc_score, y_true, y_prob,
                                   prot_attr=pa_name, priv_group=1)] )
        scores.append( ['Between-Group Balanced Accuracy Difference',
                        difference(balanced_accuracy_score, y_true, y_pred,
                                   prot_attr=pa_name, priv_group=1)] )
    else:
        pass
    scores.append( ['** Individual Measures **', None])
    scores.append( ['Consistency Score', consistency_score(X, y_pred.iloc[:,0])] )
    scores.append( ['Between-Group Generalized Entropy Error',
                        between_group_generalized_entropy_error(y_true, y_pred,
                                                            prot_attr=pa_name)])
    #
    model_scores =  pd.DataFrame(scores, columns=['Measure','Value'])
    model_scores['Value'] = model_scores.loc[:,'Value'].round(4)
    return(model_scores.fillna(""))
