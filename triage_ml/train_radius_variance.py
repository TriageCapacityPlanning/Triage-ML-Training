from .models.radius_variance import RadiusVariance
from .data.dataset import DataSet

from tensorflow.keras.optimizers import Adam


def train(dataset: DataSet, epochs: int, lr=0.001, valid_split=0.1, output_file='weights.h5'):
    rv_model = RadiusVariance(seq_size=1, radius_days=1)
    rv_model.get_model().summary()

    ml_dataset = rv_model.create_ml_dataset(dataset)

    rv_model.get_model().compile(
        optimizer=Adam(lr=lr),
        loss='mse'
    )
    rv_model.get_model().fit(
        x=ml_dataset.inputs,
        y=ml_dataset.outputs,
        validation_split=valid_split,
        epochs=epochs)

    rv_model.get_model().save(output_file)
