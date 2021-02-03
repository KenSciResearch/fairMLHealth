import aif360.sklearn.metrics as aif_mtrc
from copy import deepcopy
from fairmlhealth import model_comparison as fhmc, reports
import numpy as np
import pandas as pd
from scipy import stats
import sklearn.metrics as sk_metric
from . import __classification_metrics as clmtrc
from .utils import __preprocess_input

from .__classification_metrics import (sensitivity, specificity,
                                        false_alarm_rate, miss_rate)

def __get_ynames(df=None):
    names =  {'y': '__fairmlhealth_y_true',
              'yh': '__fairmlhealth_y_pred',
              'yp': '__fairmlhealth_y_prob'}
    if df is not None:
        for k in names.keys():
            names[k] = None if names[k] not in list(df) else names[k]
    return names


def __preprocess_stratified(X, y_true, y_pred=None, y_prob=None,
                            features:list=None):
    """
    Formats data for use in stratified reporting

    Args:
        df:
        y (string or array):
        yh (string or array):
        yp (string or array):
    Requirements:
        - Each feature must be discrete to run stratified analysis, and must be
        binary to run the assessment. If any data are not discrete and there
        are more than 11 values, the reporter will reformat those data into
        quantiles
    """
    #
    max_cats = 11
    xtiles = 5
    #
    if y_pred is None:
        X, prtc_attr, y_true, _, y_prob = \
            __preprocess_input(X, None, y_true, y_true, y_prob)
        y_pred = pd.Series(np.zeros(y_true.shape[0]))
    else:
        X, prtc_attr, y_true, y_pred, y_prob = \
            __preprocess_input(X, None, y_true, y_pred, y_prob)
    if y_prob is None:
        y_prob = pd.Series(np.zeros(y_true.shape[0]))
    #
    y, yh, yp = __get_ynames().values()
    pred_cols = [y, yh, yp]
    #
    df = deepcopy(X)
    df[y] = y_true.values
    df[yh] = y_pred.values
    df[yp] = y_prob.values
    if features is None:
        features = list(X)
    stratified_features = [f for f in features if f not in pred_cols]
    df = df.loc[:, stratified_features + pred_cols]
    #
    for f in stratified_features:
        # stratified analysis can only be run on discrete columns
        #
        if (df[f].nunique() > max_cats and
            not df[f].astype(str).str.isdigit().all()):
            print(f"\t{f} has more than {max_cats} values, which will",
                      "slow processing time. Consider reducing to quantiles")
        elif df[f].isnull().any():
            df[f].fillna(np.nan, inplace=True)
        df[f] = df[f].astype(str)
    return df


def data_report(X, y_true, features:list=None):
    """
    Generates a table of stratified data metrics

    Args:
        df:
        y_true: and must be binary

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the reporter will
        reformat those data into quantiles
    """
    #
    if features is None:
        features = list(X)
    df = __preprocess_stratified(X, y_true, features=features)
    y, yh, yp = __get_ynames(df).values()
    pred_cols = [n for n in [y, yh, yp] if n is not None]
    stratified_features = [f for f in features if f not in pred_cols]
    #
    res = []
    N_missing = 0
    N_obs = df.shape[0]
    for f in stratified_features:
        n_missing = df.loc[df[f].isnull() | df[f].eq('nan'), f].count()
        #
        grp = df.groupby(f)[pred_cols]
        # Note that the sub-functions use a cached version of
        try:
            r = grp.apply(lambda x: pd.Series(__dt_grp(x, y)))
        except BaseException as e:
            raise ValueError(f"Error processing {f}. {e}\n")
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        r.insert(0, 'FEATURE', f)
        r['N_MISSING'] = n_missing
        N_missing += n_missing
        _, feat_count = np.unique(df[f], return_counts=True)
        r['FEATURE_ENTROPY'] = stats.entropy(feat_count, base=2)
        res.append(r)
    full_res = pd.concat(res, ignore_index=True)
    full_res['VALUE_PREVALENCE'] = full_res['N OBS']/N_obs

    #
    N_feat = len(stratified_features)
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES",
                'N_MISSING': N_missing,
                'VALUE_PREVALENCE': (N_obs*N_feat - N_missing)/(N_obs*N_feat)
                }
    ov_dict = __dt_grp(df, y)
    for k, v in ov_dict.items():
        overview[k] = v
    overview_df = pd.DataFrame(overview, index=[0])
    #
    rprt = pd.concat([overview_df, full_res], axis=0, ignore_index=True)
    rprt = rprt.round(4)
    return rprt


def __dt_grp(x, y):
    _, feat_count = np.unique(y, return_counts=True)
    res = {'N OBS': x.shape[0],
           'Y_MEAN': x[y].mean(),
           'Y_SD': x[y].std(),
           'Y_MIN': x[y].min(),
           'Y_MAX': x[y].max()
           }
    return res


def regression_performance(X, y_true, y_pred, features:list=None):
    """
    Generates a table of stratified performance metrics for each specified
    feature

    Args:
        df:
        y_true: and must be binary

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the reporter will
        reformat those data into quantiles
    """
    #
    if features is None:
        features = list(X)
    df = __preprocess_stratified(X, y_true, y_pred, features=features)
    y, yh, yp = __get_ynames(df).values()
    pred_cols = [n for n in [y, yh, yp] if n is not None]
    stratified_features = [f for f in features if f not in pred_cols]
    #
    res = []
    for f in stratified_features:
        # Data are expected in string format
        assert df[f].astype(str).eq(df[f]).all()
        #
        grp = df.groupby(f)[pred_cols]
        # Note that the sub-functions use a cached version of
        try:
            r = grp.apply(lambda x: pd.Series(__rp_grp(x, y, yh)))
        except BaseException as e:
            print(f"Error processing {f}. {e}\n")
            continue
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        r.insert(0, 'FEATURE', f)
        res.append(r)
    full_res = pd.concat(res, ignore_index=True)
    #
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES"}
    ov_dict = __rp_grp(df, y, yh)
    for k, v in ov_dict.items():
        overview[k] = v
    overview_df = pd.DataFrame(overview, index=[0])
    #
    rprt = pd.concat([overview_df, full_res], axis=0, ignore_index=True)
    rprt = rprt.round(4)
    return rprt


def __rp_grp(x, y, yh):
    """ """
    res = {'N OBS': x.shape[0],
           'Y_MEAN': x[y].mean(),
            'Y_MEDIAN': x[y].median(),
            'Y_STD': x[y].std(),
            'PRED_MEAN': x[yh].mean(),
            'PRED_MEDIAN': x[yh].median(),
            'PRED_STD': x[yh].std(),
            'ERROR_MEAN': (x[yh] - x[y]).mean(),
            'ERROR_STD': (x[yh] - x[y]).std(),
            'MAE': sk_metric.mean_absolute_error(x[y], x[yh]),
            'MSE': sk_metric.mean_squared_error(x[y], x[yh]),
            'Rsqrd': sk_metric.r2_score(x[y], x[yh])}
    return res


def classification_performance(X, y_true, y_pred, y_prob=None,
                               features:list=None):
    """
    Generates a table of stratified performance metrics for each specified
    feature

    Args:
        df:
        y_true: and must be binary

    """
    #
    if features is None:
        features = list(X)
    df = __preprocess_stratified(X, y_true, y_pred, y_prob, features=features)
    y, yh, yp = __get_ynames(df).values()
    pred_cols = [n for n in [y, yh, yp] if n is not None]
    stratified_features = [f for f in features if f not in pred_cols]
    #
    res = []
    for f in stratified_features:
        # Data are expected in string format
        assert df[f].astype(str).eq(df[f]).all()
        #
        grp = df.groupby(f)[pred_cols]
        # Note that the sub-functions use a cached version of
        try:
            r = grp.apply(lambda x: pd.Series(__cp_group(x, y, yh, yp)))
        except BaseException as e:
            # raise ValueError(f"Error processing {f}. {e}\n")
            print(f"Error processing {f}. {e}\n")
            continue
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        r.insert(0, 'FEATURE', f)
        res.append(r)
    full_res = pd.concat(res, ignore_index=True)
    #
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES"}
    ov_dict = __cp_group(df, y, yh, yp)
    for k, v in ov_dict.items():
        overview[k] = v
    overview_df = pd.DataFrame(overview, index=[0])
    #
    rprt = pd.concat([overview_df, full_res], axis=0, ignore_index=True)
    rprt = rprt.round(4)
    return rprt


def __cp_group(x, y, yh, yp):
    res = {'N OBS': x.shape[0],
           'POSITIVE CLASS RATE': x[y].mean(),
            'POSITIVE PREDICTION RATE': x[yh].mean(),
            'ACCURACY': clmtrc.accuracy(x[y], x[yh]),
            'PRECISION': clmtrc.precision(x[y], x[yh]), #PPV
            'TPR': clmtrc.sensitivity(x[y], x[yh]),
            'TNR': clmtrc.specificity(x[y], x[yh]),
            'FPR': clmtrc.false_alarm_rate(x[y], x[yh]),
            'FNR': clmtrc.miss_rate(x[y], x[yh])
            }
    if yp is not None:
        res['ROC_AUC'] = clmtrc.roc_auc_score(x[y], x[yp])
        res['PR_AUC'] = clmtrc.pr_auc_score(x[y], x[yp])
    return res


def regression_fairness(X, y_true, y_pred, features: list = None, **kwargs):
    """
    Generates a table of stratified fairness metrics metrics for each specified
    feature

    Args:
        df:
        y_true: and must be binary

    """
    #
    if features is None:
        features = list(X)
    df = __preprocess_stratified(X, y_true, y_pred, features=features)
    y, yh, yp = __get_ynames(df).values()
    pred_cols = [n for n in [y, yh, yp] if n is not None]
    stratified_features = [f for f in features if f not in pred_cols]
    #
    res = []
    for f in stratified_features:
        # Data are expected in string format
        assert df[f].astype(str).eq(df[f]).all()
        #
        grp = df.groupby(f, as_index=False)[y].count()
        grp.rename(columns={f: 'FEATURE VALUE', y: 'N OBS'}, inplace=True)
        grp['FEATURE'] = f
        res.append(grp)
    rprt = pd.concat(res, axis=0, ignore_index=True)
    #
    res_f = []
    rprt = rprt[['FEATURE', 'FEATURE VALUE', 'N OBS']]
    pa_name = 'prtc_attr'
    for i, row in rprt.iterrows():
        f = row['FEATURE']
        v = row['FEATURE VALUE']
        df[pa_name] = 0
        df.loc[df[f].eq(v), pa_name] = 1
        if v != "nan":
            df.loc[df[f].eq("nan"), pa_name] = np.nan
        try:
            subset = df.loc[df[pa_name].notnull(),
                            [pa_name, y, yh]].set_index(pa_name)
            meas = __rf_group(pa_name, subset[y], subset[yh], priv_grp=1)
        except BaseException as e:
            # raise ValueError(f"Error processing {f}. {e}\n")
            print(f"Error processing {f}. {e}\n")
            continue
        r = pd.DataFrame(meas, index=[0])
        r['FEATURE'] = f
        r['FEATURE VALUE'] = v
        res_f.append(r)
    rprt_update = pd.concat(res_f, ignore_index=True)
    rprt_update = rprt_update[sorted(rprt_update.columns, key=lambda x: x[-5:])]
    rprt = rprt.merge(rprt_update, on=['FEATURE', 'FEATURE VALUE'], how='left')
    rprt = rprt.round(4)
    #
    return rprt


def __rf_group(pa_name, y_true, y_pred, priv_grp=1):
    res = reports.__regres_group_fairness_measures(pa_name, y_true, y_pred,
                                                   priv_grp)
    return res


def classification_fairness(X, y_true, y_pred, features:list=None, **kwargs):
    """
    Generates a table of stratified fairness metrics metrics for each specified
    feature

    Args:
        df:
        y_true: and must be binary

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the reporter will
        reformat those data into quantiles
    """
    #
    if features is None:
        features = list(X)
    df = __preprocess_stratified(X, y_true, y_pred, features=features)
    y, yh, yp = __get_ynames(df).values()
    pred_cols = [n for n in [y, yh, yp] if n is not None]
    stratified_features = [f for f in features if f not in pred_cols]
    #
    res = []
    pa_name = 'prtc_attr'
    for f in stratified_features:
        vals = sorted(df[f].unique().tolist())
        for v in vals:
            df[pa_name] = 0
            df.loc[df[f].eq(v), pa_name] = 1
            if v != "nan":
                df.loc[df[f].eq("nan"), pa_name] = np.nan
            try:
                subset = df.loc[df[pa_name].notnull(),
                                [pa_name, y, yh]].set_index(pa_name)
                meas = __cf_group(pa_name, subset[y], subset[yh], priv_grp=1)
            except BaseException as e:
                # raise ValueError(f"Error processing {f}. {e}\n")
                print(f"Error processing {f}. {e}\n")
                continue
            r = pd.DataFrame(meas, index=[0])
            r['N OBS'] =  df.loc[df[f].eq(v), pa_name].sum()
            r['FEATURE'] = f
            r['FEATURE VALUE'] = v
            res.append(r)
    full_res = pd.concat(res, ignore_index=True)
    id_cols = ['FEATURE', 'FEATURE VALUE']
    full_res = full_res[id_cols + [c for c in list(full_res) if c not in id_cols]]
    #
    rprt = full_res.round(4)
    #
    return rprt


def __cf_group(pa_name, y_true, y_pred, priv_grp=1):
    """ """
    gf_vals = {}
    gf_vals['PPV Ratio'] = \
        aif_mtrc.difference(sk_metric.precision_score, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TPR Ratio'] = \
        aif_mtrc.ratio(clmtrc.sensitivity, y_true, y_pred, prot_attr=pa_name,
                       priv_group=priv_grp)
    gf_vals['FPR Ratio'] = \
        aif_mtrc.ratio(clmtrc.false_alarm_rate, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TNR Ratio'] = \
        aif_mtrc.ratio(clmtrc.specificity, y_true, y_pred, prot_attr=pa_name,
                       priv_group=priv_grp)
    gf_vals['FNR Ratio'] = \
        aif_mtrc.ratio(clmtrc.miss_rate, y_true, y_pred, prot_attr=pa_name,
                       priv_group=priv_grp)
    #
    gf_vals['PPV Difference'] = \
        aif_mtrc.difference(sk_metric.precision_score, y_true,
                            y_pred, prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TPR Difference'] = \
        aif_mtrc.difference(clmtrc.sensitivity, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['FPR Difference'] = \
        aif_mtrc.difference(clmtrc.false_alarm_rate, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TNR Difference'] = \
        aif_mtrc.difference(clmtrc.specificity, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['FNR Difference'] = \
        aif_mtrc.difference(clmtrc.miss_rate, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    return gf_vals
