"""
"""
# ToDo: Add more robust testing throughout


from fairmlhealth import stat_utils
import numpy as np
import pytest
import pandas as pd
from .__utils import synth_dataset


@pytest.fixture(scope="class")
def load_classification_data(request):
    N = 32
    df = synth_dataset(N)
    X = df[["A", "B", "C", "D", "prtc_attr", "other"]]
    y = df["binary_target"].rename("y")
    avg_fair = df["avg_binary_pred"].rename("avg")
    #
    data = pd.concat([X, y, avg_fair], axis=1)
    rng = np.random.RandomState(506)
    cohorts = pd.DataFrame({0: rng.randint(0, 2, N), 1: rng.randint(0, 2, N)})
    #
    request.cls.df = data
    request.cls.cohorts = cohorts
    yield


@pytest.mark.usefixtures("load_classification_data")
class TestBootstrapping:
    """ Validates that standard inputs are processed without error
    """

    def test_true(self):
        assert stat_utils.bootstrap_significance(
            func=stat_utils.kruskal_pval, a=self.df["A"], b=self.df["B"]
        )

    def test_false(self):
        assert not stat_utils.bootstrap_significance(
            func=stat_utils.kruskal_pval, a=self.df["A"], b=self.df["A"]
        )
