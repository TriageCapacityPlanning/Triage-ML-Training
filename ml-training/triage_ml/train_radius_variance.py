from triage_ml.models.radius_variance import RadiusVariance
from triage_ml.data.dataset import DataSet

from tensorflow.keras.optimizers import Adam
from collections import namedtuple


def train(dataset: DataSet, epochs: int, lr=0.001, valid_split=0.1, output_file='weights.h5'):
    rv_model = RadiusVariance(seq_size=30, radius_days=15)
    rv_model.get_model().summary()

    ml_dataset = rv_model.create_ml_dataset(dataset)
    train_data, test_data = ml_dataset.split(1 - valid_split)

    rv_model.get_model().compile(
        optimizer=Adam(lr=lr),
        loss='mse'
    )

    history = rv_model.get_model().fit(
        x=train_data.inputs,
        y=train_data.outputs,
        validation_data=(test_data.inputs, test_data.outputs),
        epochs=epochs)

    rv_model.get_model().save(output_file)
    return rv_model, train_data, test_data, history
