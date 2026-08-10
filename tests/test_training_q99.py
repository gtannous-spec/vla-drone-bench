import numpy as np
from airsim_benchmark.core.action_space import VLN_Q99 as SQ
from airsim_benchmark.training.train_regression import VLN_Q99 as TQ
from airsim_benchmark.training.airsim_dataset import VLN_Q99 as DQ


def test_training_q99_matches_shared():
    assert np.allclose(TQ, SQ)
    assert np.allclose(DQ, SQ)
    assert SQ[5] == 2.0
