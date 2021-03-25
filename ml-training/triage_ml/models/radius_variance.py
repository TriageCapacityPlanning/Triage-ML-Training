from triage_ml.models.prediction_model import PredictionModel
from triage_ml.data.dataset import DataSet, MLDataSet, TimeInterval

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

    def __init__(self, seq_size=30, radius=15, time_interval=TimeInterval.DAY):
        """
        Construct a new RadiusVariance.
        :param seq_size: How many previous days does the model require to predict.
        :param radius: The radius in time_interval units used to compute arrival variance.
        :param time_interval: The size of each unit time interval.
        """
        self.seq_size = seq_size
        self.radius = radius
        self.time_interval = time_interval
        self.model = None

    def _init_model(self):
        seq_input = Input(shape=(self.seq_size, 1), name='seq_input')
        date_input = Input(shape=(12 + 31), name='date_input')  # One-hot encoding of month and day of month

        x = LSTM(units=8, activation='tanh')(seq_input)
        x = Dropout(0.2)(x)

        x = Concatenate(axis=1)([x, date_input])

        x = Dense(units=64, activation='tanh')(x)
        x = Dropout(0.2)(x)
        """
        x = Dense(units=512, activation='tanh')(x)
        x = Dropout(0.2)(x)
        x = Dense(units=256, activation='tanh')(x)
        x = Dropout(0.2)(x)
        """

        output = Dense(units=2, activation='relu')(x)

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
            min_date = first.date_received + self._timedelta(self.radius)
            max_date = last.date_received - self._timedelta(self.radius)

            if self.time_interval == TimeInterval.WEEK:
                date_aggregation = dataset.aggregate_on('date_received', lambda dr: self._datestr(dr))
            else:
                date_aggregation = dataset.aggregate_on('date_received', lambda dr: self._datestr(dr))

            date_range = list(self._date_range(min_date, max_date))
        else:
            date_range = []

        x = [np.zeros((len(date_range), 1)), np.zeros((len(date_range), 12 + 31))]
        y = [np.zeros((len(date_range), 2))]

        for i, date in enumerate(date_range):
            date_str = self._datestr(date)
            val = len(date_aggregation[date_str]) if date_str in date_aggregation else 0

            radius_vals = []
            for j in range(-self.radius, self.radius + 1):
                j_date_str = self._datestr(date + self._timedelta(j))
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
        if self.time_interval == TimeInterval.WEEK:
            size = int((max_date - min_date).days / 7) + 1
            delta = timedelta(weeks=1)
        else:
            size = int((max_date - min_date).days) + 1
            delta = timedelta(days=1)

        for n in range(size):
            yield min_date + n*delta

    def _timedelta(self, amount):
        if self.time_interval == TimeInterval.WEEK:
            return timedelta(weeks=amount)
        else:
            return timedelta(days=amount)

    def _datestr(self, date):
        if self.time_interval == TimeInterval.WEEK:
            return datetime.strftime(date, '%Y-%U')
        else:
            return str(date.date())

    def get_model(self):
        """
        :return: The Keras model.
        """
        if self.model is None:
            self._init_model()
            return self.model
        else:
            return self.model

