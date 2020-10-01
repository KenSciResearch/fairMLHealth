from abc import ABC
import aif360.sklearn.metrics as aif_mtrc
import fairlearn.metrics as fl_mtrc
import fairMLHealth.utils.measures as fh_mtrc
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metric
import warnings
# Temporarily hide pandas SettingWithCopy warning
warnings.filterwarnings('ignore', module='pandas' )
warnings.filterwarnings('ignore', module='sklearn' )



'''
    Global variable to turn off 
'''
TUTORIAL_ON = False

def start_tutorial():
    global TUTORIAL_ON
    TUTORIAL_ON = True

def stop_tutorial():
    global TUTORIAL_ON
    TUTORIAL_ON = True

def is_tutorial_running():
    return TUTORIAL_ON



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

def __validate_report_inputs(X, protected_attr, y_true, y_pred, y_prob=None):
    """[summary]
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


def format_comparison_inputs(X, protected_attr, y_true, y_pred, y_prob=None):
    """

    Args:
        X ([type]): [description]
        protected_attr ([type]): [description]
        y_true ([type]): [description]
        y_pred ([type]): [description]
        y_prob ([type], optional): [description]. Defaults to None.

    Raises:
        TypeError: [description]
    """
    __validate_report_inputs(X, protected_attr, y_true, y_pred, y_prob)

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
        if data is not None and data.shape[1] > 1:
            raise TypeError("targets and predictions must be 1-Dimensional")

    # Format and set sensitive attributes as index for y dataframes
    pa_name = protected_attr.columns.tolist()
    protected_attr.reset_index(inplace=True, drop=True)
    y_true = pd.concat([protected_attr, y_true.reset_index(drop=True)], axis=1
                       ).set_index(pa_name)
    y_pred = pd.concat([protected_attr, y_pred.reset_index(drop=True)], axis=1
                       ).set_index(pa_name)
    y_pred.columns = y_true.columns
    if y_prob is not None:
        y_prob = pd.concat([protected_attr, y_prob.reset_index(drop=True)], axis=1
                        ).set_index(pa_name)
        y_prob.columns = y_true.columns
    return(X, protected_attr, y_true, y_pred, y_prob, pa_name)



def report_classification_fairness(X, protected_attr, y_true, y_pred, y_prob=None):
    """ Returns a dataframe containing fairness measures for the model results

        Args:
            X (array-like): Sample features
            protected_attr (array-like, named): values for the protected attribute
                (note: protected attribute may also be present in X)
            y_true (array-like, 1-D): Sample targets
            y_pred (array-like, 1-D): Sample target predictions
            y_prob (array-like, 1-D): Sample target probabilities

        Returns:
            pandas dataframe
    """
    X, prtc_attr, y_true, y_pred, y_prob, pa_names = \
        format_comparison_inputs(X, protected_attr, y_true, y_pred, y_prob)

    # Ensure that protected_attr is integer-valued
    pa_cols = prtc_attr.columns.tolist()
    for c in pa_cols:
        binary = ( set(prtc_attr[c].astype(int)) == set(prtc_attr[c]) )
        boolean = ( prtc_attr[c].dtype == bool )
        two_valued = ( set(prtc_attr[c].astype(int)) == {0,1} )
        if not two_valued and (binary or boolean):
            raise ValueError(
            "prtc_attr must be binary or boolean and heterogeneous")
        prtc_attr.loc[:, c] = prtc_attr[c].astype(int)
        if isinstance(c, int):
            prtc_attr.rename(columns={c:f"prtc_attribute_{c}"}, inplace=True)

    # Generate lists of performance measures to be converted to dataframe
    scores = {}
    n_class = y_true.append(y_pred).iloc[:,0].nunique()

    # Generate dict of group fairness measures, if applicable
    if n_class == 2:
        gf_vals = {}
        gf_key = '** Group Fairness **'
        gf_vals['Statistical Parity Difference'] = \
                aif_mtrc.statistical_parity_difference(y_true, y_pred,
                                                        prot_attr=pa_names)
        gf_vals['Disparate Impact Ratio'] = \
                aif_mtrc.disparate_impact_ratio(y_true, y_pred,
                                                prot_attr=pa_names)
        if not is_tutorial_running():
            gf_vals['Demographic Parity Difference'] = \
                fl_mtrc.demographic_parity_difference(y_true, y_pred,
                                            sensitive_features=prtc_attr)
            gf_vals['Demographic Parity Ratio'] = \
                fl_mtrc.demographic_parity_ratio(y_true, y_pred,
                                            sensitive_features=prtc_attr)
        gf_vals['Average Odds Difference'] = \
                aif_mtrc.average_odds_difference(y_true, y_pred,
                                                    prot_attr=pa_names)
        gf_vals['Equal Opportunity Difference'] = \
                aif_mtrc.equal_opportunity_difference(y_true, y_pred,
                                                        prot_attr=pa_names)
        if not is_tutorial_running():
            gf_vals['Equalized Odds Difference'] = \
                fl_mtrc.equalized_odds_difference(y_true, y_pred,
                                            sensitive_features=prtc_attr)
            gf_vals['Equalized Odds Ratio'] = \
                fl_mtrc.equalized_odds_ratio(y_true, y_pred,
                                            sensitive_features=prtc_attr)
        gf_vals['Positive Predictive Parity Difference'] = \
                aif_mtrc.difference(sk_metric.precision_score, y_true,
                                    y_pred, prot_attr=pa_names, priv_group=1)
        gf_vals['Balanced Accuracy Difference'] = \
                aif_mtrc.difference(sk_metric.balanced_accuracy_score, y_true,
                                     y_pred, prot_attr=pa_names, priv_group=1)
        if y_prob is not None:
            gf_vals['AUC Difference'] = \
                aif_mtrc.difference(sk_metric.roc_auc_score, y_true, y_prob,
                                     prot_attr=pa_names, priv_group=1)
    #
    if_vals, if_key = __individual_fairness_measures(X, y_true, y_pred, pa_names)

    # Generate a model performance report
    # If more than 2 classes, return the weighted average prediction scores
    target_labels = [f"target = {t}" for t in set(np.unique(y_true))]
    report = classification_report(y_true.iloc[:,0], y_pred.iloc[:,0], target_labels)
    avg_lbl = "weighted avg" if n_class > 2 else target_labels[-1]
    #
    mp_vals = {}
    c_note = "" if n_class == 2 else "(Weighted Avg)"
    mp_key = f'** Model Performance {c_note}**'
    for score in ['precision', 'recall', 'f1-score']:
        mp_vals[score.title()] = report.loc[avg_lbl, score]
    mp_vals['Accuracy'] = report.loc['accuracy', 'accuracy']

    # Convert scores to a formatted dataframe and return
    measures = {gf_key:gf_vals,if_key:if_vals, mp_key: mp_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df['Value'] = df.loc[:,'Value'].round(4)
    df.fillna("", inplace=True)
    return df



def __individual_fairness_measures(X, y_true, y_pred, pa_names):
    """
    """
    # Generate dict of Individual Fairness measures
    if_vals = {}
    if_key = '** Individual Fairness **'
    if_vals['Consistency Score'] = \
            aif_mtrc.consistency_score(X, y_pred.iloc[:,0])
    if_vals['Between-Group Generalized Entropy Error'] = \
            aif_mtrc.between_group_generalized_entropy_error(y_true, y_pred,
                                                              prot_attr=pa_names)
    return if_vals, if_key


def classification_report(y_true, y_pred, target_labels=None):
    """ Returns a formatted dataframe of the scikit-learn classification report

    Args:
        y_true (scikit-compatible array): target values
        y_pred (scikit-compatible array): prediction values
        target_labels (list of str): optional labels to elucidate target values
    """
    if target_labels is None:
        target_labels = [f"target = {t}" for t in set(y_true)]
    report = sk_metric.classification_report(y_true, y_pred, output_dict=True,
                                            target_names=target_labels)
    report = pd.DataFrame(report).transpose()
    # Move accuracy to separate row
    accuracy = report.loc['accuracy',:]
    report.drop('accuracy', inplace=True)
    report.loc['accuracy', 'accuracy'] = accuracy[0]
    return(report)


def report_regression_fairness(X, protected_attr, y_true, y_pred):
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
    X, prtc_attr, y_true, y_pred, _, pa_names = \
        format_comparison_inputs(X, protected_attr, y_true, y_pred)
    gf_vals = {}
    gf_key = '** Group Fairness **'
    gf_vals['Statistical Parity Ratio'] = \
            fh_mtrc.statistical_parity_ratio(y_true, y_pred,
                                              prot_attr=prtc_attr)
    gf_vals['R2 Ratio'] = \
            aif_mtrc.ratio(sk_metric.r2_score, y_true, y_pred,
                           prot_attr=pa_names, priv_group=1)
    gf_vals['MAE Ratio'] = \
            aif_mtrc.ratio(sk_metric.mean_absolute_error, y_true, y_pred,
                           prot_attr=pa_names, priv_group=1)
    gf_vals['MSE Ratio'] = \
            aif_mtrc.ratio(sk_metric.mean_squared_error, y_true, y_pred,
                           prot_attr=pa_names, priv_group=1)
    #
    if_vals, if_key = __individual_fairness_measures(X, y_true, y_pred, pa_names)
    #
    mp_vals = {}
    mp_key = '** Model Performance **'
    report = regression_report(y_true, y_pred)
    for row in report.iterrows():
        mp_vals[row[0]] = row[1]['Score']
    # Convert scores to a formatted dataframe and return
    measures = {gf_key:gf_vals,if_key:if_vals, mp_key: mp_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df['Value'] = df.loc[:,'Value'].round(4)
    df.fillna("", inplace=True)
    return df



def regression_report(y_true, y_pred):
    """ Returns a report of the regression metrics, similar to scikit's
        classification_report

    Args:
        y_true (scikit-compatible array): target values
        y_pred (scikit-compatible array): prediction values
    """
    report = {}
    report['Rsqrd'] = sk_metric.r2_score(y_true, y_pred)
    report['MeanAE'] = sk_metric.mean_absolute_error(y_true, y_pred)
    report['MeanSE'] = sk_metric.mean_squared_error(y_true, y_pred)
    report = pd.DataFrame().from_dict(report, orient='index'
                          ).rename(columns={0:'Score'})
    return(report)


def highlight_suspicious_scores(df, caption="", model_type="binary"):
    """ Returns a pandas styler table containing a hilighted version of a
        model comparison dataframe

    Args:
        df (pandas dataframe): model comparison dataframe as output by
            report_regression_fairness or report_classification_fairness
        caption (str, optional): Optional caption for the table. Defaults to "".

    Returns:
        pandas.io.formats.style.Styler
    """
    if caption is None:
        caption = "Fairness Meaures"
    #
    idx = pd.IndexSlice
    measures = df.index.get_level_values(1)
    ratios = df.loc[idx['** Group Fairness **',
                    [c.lower().endswith("ratio") for c in measures]],:].index
    difference = df.loc[idx['** Group Fairness **',
                    [c.lower().endswith("difference") for c in measures]],:].index
    cs = df.loc[idx['** Group Fairness **',
                    [c.lower().replace(" ", "_") == "consistency_score" for c in measures]],:].index

    return(df.style.set_caption(caption
            ).apply(lambda x: ['color:orange'
                    if (x.name in ratios and not 1 < x.iloc[0] < 1.2)
                    else '' for i in x], axis=1
            ).apply(lambda x: ['color:orange'
                    if (x.name in difference and not -0.1 < x.iloc[0] < 0.1)
                    else '' for i in x], axis=1
            ).apply(lambda x: ['color:orange'
                    if (x.name in cs and x.iloc[0] < 0.5) else '' for i in x], axis=1
            )
    )