from triage_ml.models.radius_variance import RadiusVariance
from triage_ml.data.dataset import DataSet

import pytest
from datetime import datetime
import numpy as np
from tensorflow.keras.optimizers import Adam


@pytest.fixture(scope="module")
def test_dataset():
    data = []
    with open('tests/test_data.txt') as f:
        for line in f.readlines():
            line = line.rstrip().split(',')
            data.append((
                int(line[0]),
                int(line[1]),
                datetime.strptime(line[2], '%Y-%m-%d'),
            ))

    return DataSet(data)


def test_get_model():
    model = RadiusVariance()
    model.get_model().summary()


def test_train_model(test_dataset):
    """SRS: MOD-2"""
    model = RadiusVariance(seq_size=2, radius_days=1)
    ml_dataset = model.create_ml_dataset(test_dataset)
    weights = model.get_model().get_weights()

    model.get_model().compile(
        optimizer=Adam(lr=0.001),
        loss='mse'
    )
    model.get_model().fit(x=ml_dataset.inputs, y=ml_dataset.outputs, epochs=1)

    new_weights = model.get_model().get_weights()

    assert not np.all([np.array_equal(w, new_w) for (w, new_w) in zip(weights, new_weights)])


def test_prediction(test_dataset):
    """SRS: MOD-7"""
    model = RadiusVariance(seq_size=2, radius_days=1)
    ml_dataset = model.create_ml_dataset(test_dataset)
    model.get_model().compile(
        optimizer=Adam(lr=0.001),
        loss='mse'
    )

    y_hat = model.get_model().predict([x[np.newaxis, 0] for x in ml_dataset.inputs])

    assert y_hat.shape == (1, 2)


def test_prediction_batch(test_dataset):
    model = RadiusVariance(seq_size=2, radius_days=1)
    ml_dataset = model.create_ml_dataset(test_dataset)
    model.get_model().compile(
        optimizer=Adam(lr=0.001),
        loss='mse'
    )

    y_hat = model.get_model().predict([x[0:2] for x in ml_dataset.inputs])

    assert y_hat.shape == (2, 2)


def test_create_ml_dataset(test_dataset):
    model = RadiusVariance(seq_size=2, radius_days=3)
    ml_dataset = model.create_ml_dataset(test_dataset)

    assert np.array_equal(ml_dataset.inputs[0], np.array([[[1], [1]], [[1], [0]], [[0], [4]]]))


def test_create_ml_dataset_empty():
    model = RadiusVariance(seq_size=3, radius_days=2)
    dataset = DataSet([])
    ml_dataset = model.create_ml_dataset(dataset)

    assert np.array_equal(ml_dataset.inputs[0], np.array([]))
    assert np.array_equal(ml_dataset.inputs[1], np.empty((0, 43)))
    assert np.array_equal(ml_dataset.outputs[0], np.empty((0, 2)))


def test_create_ml_dataset_correct_length(test_dataset):
    model = RadiusVariance(seq_size=1, radius_days=1)
    dataset = DataSet(test_dataset)
    ml_dataset = model.create_ml_dataset(dataset)

    assert ml_dataset.inputs[0].shape[0] == 8


def test_create_ml_dataset_radius_affects_length(test_dataset):
    model = RadiusVariance(seq_size=3, radius_days=3)
    dataset = DataSet(test_dataset)
    ml_dataset = model.create_ml_dataset(dataset)

    assert ml_dataset.inputs[0].shape[0] == 2
