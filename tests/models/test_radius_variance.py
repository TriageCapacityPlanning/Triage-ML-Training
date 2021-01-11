from triage_ml.models.radius_variance import RadiusVariance
from triage_ml.data.dataset import DataSet

import pytest
from datetime import datetime
import numpy as np


@pytest.fixture(scope="module")
def test_dataset():
    data = []
    with open('tests/test_data.txt') as f:
        for line in f.readlines():
            line = line.rstrip().split(',')
            data.append((
                int(line[0]),
                int(line[1]),
                int(line[2]),
                datetime.strptime(line[3], '%Y-%m-%d'),
                datetime.strptime(line[4], '%Y-%m-%d')
            ))

    return DataSet(data)


def test_get_model():
    model = RadiusVariance()
    model.get_model().summary()


def test_create_ml_dateset(test_dataset):
    model = RadiusVariance(radius_days=1)
    ml_dataset = model.create_ml_dataset(test_dataset)

    assert list(ml_dataset.inputs[0]) == [0, 2, 1, 1, 0, 4, 0, 0, 0]
