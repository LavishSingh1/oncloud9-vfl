import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    from vfl.data import prepare_vertical_dataset
    d = tmp_path_factory.mktemp("data")
    prepare_vertical_dataset(str(d), "breast_cancer", n_passive=1, overlap=0.9, seed=0)
    return str(d)


@pytest.fixture
def cfg():
    from vfl.config import load_session_config
    return load_session_config(None, ["epochs=2", "batch_size=32"])
