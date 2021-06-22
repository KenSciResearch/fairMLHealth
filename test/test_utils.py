'''
'''
 # ToDo: Add more robust testing throughout


from fairmlhealth import reports
import numpy as np
import pytest
import pandas as pd
from .__utils import synth_dataset
np.random.seed(547)



@pytest.fixture(scope="class")
def load_classification_data(synth_dataset, request):
    df = synth_dataset
    targ_data = [0, 0, 1, 1, 0, 0, 1, 1]
    y = pd.Series(targ_data, index=df.index, name="y")
    avg_fair = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=df.index, name="avg")
    #

    df = pd.concat([df, y, avg_fair], axis=1)
    df = df.append(df).reset_index(drop=True)
    cohorts = pd.Series(np.random.randint(0, 2, 16), index=df.index, name="coh")
    request.cls.df = df
    request.cls.cohorts = cohorts
    yield



@pytest.mark.usefixtures("load_classification_data")
class TestCohorts:
    """ Validates that standard inputs are processed without error
    """
    def test_no_cohort(self):
        _ = reports.summary_report(self.df, self.df['prtc_attr'], self.df['y'],
                                   self.df['avg'], pred_type="classification")

    def test_one_cohort(self):
        _ = reports.summary_report(self.df, self.df['prtc_attr'], self.df['y'],
                                   self.df['avg'], pred_type="classification",
                                   cohorts=self.cohorts)

    def test_cohort_stratified(self):
        _ = reports.bias_report(self.df, self.df['prtc_attr'], self.df['y'],
                                pred_type="classification", cohorts=self.cohorts)
