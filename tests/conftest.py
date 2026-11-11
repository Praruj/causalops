from functools import partial

import numpy as np
import pytest

from causalops.data import sample
from causalops.model import predict, train_model


@pytest.fixture(scope="session")
def world():
    """One trained model and a reference set, shared by the slower tests."""
    rng = np.random.default_rng(0)
    X_train, y_train = sample(30000, rng)
    ref_X, ref_y = sample(20000, rng)
    model = train_model(X_train, y_train, 0)
    return ref_X, ref_y, partial(predict, model)
