import pytest
import pandas as pd
import os

# Path to your cleaned data file
DATA_PATH = "data/raw_data.csv"
REF_DATA_PATH = "data/raw_data.csv"

  # 

@pytest.fixture(scope='session')
def data():
    if not os.path.exists(DATA_PATH):
        pytest.fail(f"Data file not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

@pytest.fixture(scope='session')
def ref_data():
    if not os.path.exists(REF_DATA_PATH):
        pytest.fail(f"Reference data file not found at {REF_DATA_PATH}")
    return pd.read_csv(REF_DATA_PATH)

@pytest.fixture(scope='session')
def kl_threshold():
    return 0.2   # default threshold used in Udacity rubric

@pytest.fixture(scope='session')
def min_price():
    return 10.0  # adjust if your project uses different bounds

@pytest.fixture(scope='session')
def max_price():
    return 350.0  # adjust if your project uses different bounds
