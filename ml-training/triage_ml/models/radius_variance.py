from triage_ml.models.prediction_model import PredictionModel
from triage_ml.data.dataset import DataSet, MLDataSet

from typing import List
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Concatenate, Dense
from datetime import datetime, timedelta
import numpy as np


class RadiusVariance(PredictionModel):
    """
    Predicts the number of patients arrivals and the variance of arrivals in a radius.

    Takes in the previous seq_size days predicts:
     - The number of arrivals on the next day
     - The variance of arrivals

     The variance of arrivals is trained by taking the variance of arrivals within a radius_days
     radius of the prediction day.
    """

    def __init__(self, seq_size=30, radius_days=15):
        """
        Construct a new RadiusVariance.
        :param seq_size: How many previous days does the model require to predict.
        :param radius_days: The radius in days used to compute arrival variance.
        """
        self.seq_size = seq_size
        self.radius_days = radius_days
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

    def create_ml_dataset(self, dataset: DataSet) -> MLDataSet:
        """
        Build a MLDataSet compatible with RadiusVariance using the provided DataSet.
        :param dataset: The DataSet to construct the MLDataSet from.
        :return: The MLDataSet
        """
        if len(dataset):
            dataset.order_by('date_received')
            first = dataset[0]
            last = dataset[-1]
            min_date = first.date_received + timedelta(days=self.radius_days)
            max_date = last.date_received - timedelta(days=self.radius_days)

            date_aggregation = dataset.aggregate_on('date_received', lambda dr: str(dr.date()))

            date_range = list(self._date_range(min_date, max_date))
        else:
            date_range = []

        x = [np.zeros((len(date_range), 1)), np.zeros((len(date_range), 12 + 31))]
        y = [np.zeros((len(date_range), 2))]

        for i, date in enumerate(date_range):
            date_str = str(date.date())
            val = len(date_aggregation[date_str]) if date_str in date_aggregation else 0

            radius_vals = []
            for j in range(-self.radius_days, self.radius_days + 1):
                j_date_str = str((date + timedelta(days=j)).date())
                radius_vals.append(len(date_aggregation[j_date_str]) if j_date_str in date_aggregation else 0)
            variance = np.var(radius_vals)

            one_hot_date = np.zeros(12 + 31)
            one_hot_date[date.date().month] = 1
            one_hot_date[11 + date.date().day] = 1

            x[0][i] = val
            x[1][i] = one_hot_date
            y[0][i] = [val, variance]

        return MLDataSet(x[0:1], x[1:], y, self.seq_size)

    def predict(self, seed_data: List[np.ndarray], date_encodings) -> List[np.ndarray]:
        """
        Predict one of more times.
        :param seed_data: The data to seed the model with.
        :param date_encodings: The encodings of the dates to predict on
        :return: A list of predictions.
        """
        predictions = []
        for date in date_encodings:
            seed_data[1] = date[np.newaxis, :]

            pred = self.model(seed_data)

            predictions.append(pred)
            seed_data[0][0] = seed_data[0][0][:1] + [[pred[0][0]]]

        return predictions

    def _date_range(self, min_date: datetime, max_date: datetime):
        for n in range(int((max_date - min_date).days) + 1):
            yield min_date + timedelta(n)

    def get_model(self):
        """
        :return: The Keras model.
        """
        if self.model is None:
            self._init_model()
            return self.model
        else:
            return self.model

