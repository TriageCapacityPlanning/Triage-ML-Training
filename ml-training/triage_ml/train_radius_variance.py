from triage_ml.models.radius_variance import RadiusVariance
from triage_ml.data.dataset import DataSet, TimeInterval

from tensorflow.keras.optimizers import Adam
from collections import namedtuple
import tensorflow as tf


def loss(y_true, y_pred):
    return (y_true[0] - y_pred[0])**2 \
           + (y_true[1] - y_pred[1])**2 \
           + 10 * tf.math.maximum(y_true[0] - y_pred[0], 0)**2 \
           + 10 * tf.math.maximum(y_true[1] - y_pred[1], 0)**2


def train(dataset: DataSet, epochs: int, lr=0.001, valid_split=0.2, output_file='weights.h5'):
    rv_model = RadiusVariance(seq_size=30, radius=15, time_interval=TimeInterval.WEEK)
    rv_model.get_model().summary()

    ml_dataset = rv_model.create_ml_dataset(dataset)
    train_data, test_data = ml_dataset.split(1 - valid_split)
    print(len(train_data.inputs[0]), len(test_data.inputs[0]))

    rv_model.get_model().compile(
        optimizer=Adam(lr=lr),
        loss=loss
    )

    history = rv_model.get_model().fit(
        x=train_data.inputs,
        y=train_data.outputs,
        validation_data=(test_data.inputs, test_data.outputs),
        epochs=epochs)

    rv_model.get_model().save(output_file)
    return rv_model, train_data, test_data, history
