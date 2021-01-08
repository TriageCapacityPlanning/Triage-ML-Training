from .prediction_model import PredictionModel
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Concatenate, Dense


class RadiusVariance(PredictionModel):

    def __init__(self, seq_size=30):
        self.seq_size = seq_size
        self.model = None

    def _init_model(self):
        seq_input = Input(shape=(self.seq_size, 1), name='seq_input')
        date_input = Input(shape=(12 + 31), name='date_input')  # One-hot encoding of month and day of month

        x = LSTM(units=64)(seq_input)
        x = Dropout(0.2)(x)

        x = Concatenate(axis=1)([x, date_input])

        x = Dense(units=64)(x)
        x = Dense(units=32)(x)

        output = Dense(units=2)(x)

        self.model = Model(name='radius_variance', inputs=[seq_input, date_input], outputs=output)

    def create_dataset(self, data):
        pass

    def save_dataset(self, dataset):
        pass

    def load_dataset(self, dataset_file):
        pass

    def get_model(self):
        if self.model is None:
            self._init_model()
            return self.model
        else:
            return self.model

