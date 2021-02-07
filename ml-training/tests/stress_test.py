from triage_ml.data.dataset import DataSet
from triage_ml.models.radius_variance import RadiusVariance

from datetime import datetime, timedelta
from tensorflow.keras.optimizers import Adam
import time


def date_range(min_date: datetime, max_date: datetime):
    for n in range(int((max_date - min_date).days) + 1):
        yield min_date + timedelta(n)


def create_data(start_date, end_date):
    data = []
    for index, date in enumerate(date_range(start_date, end_date)):
        data.append((index, 1, 0, date, date))
    return data


def run(years=1, year_target=5, epochs=100, epochs_target=500):
    print(f'Generating {years} year(s) of data...')
    data_start = time.time()

    data = create_data(datetime(2020 - years, 1, 1), datetime(2020, 1, 1))
    dataset = DataSet(data)

    print('Data generation done')
    print(f'Took {time.time() - data_start}s\n')

    model = RadiusVariance(seq_size=30, radius_days=15)
    ml_dataset = model.create_ml_dataset(dataset)

    print(f'Training for {epochs} epochs...')
    train_start = time.time()
    model.get_model().compile(
        optimizer=Adam(lr=0.001),
        loss='mse'
    )
    model.get_model().fit(x=ml_dataset.inputs, y=ml_dataset.outputs, epochs=epochs)
    print('Training done')
    train_time = int(time.time() - train_start)
    print(f'Took {train_time}s\n')
    total_time = train_time * (year_target / years) * (epochs_target / epochs)

    print(f'It would take this machine an estimated {int(total_time // 60)}m{int(total_time % 60)}s '
          + f'to train {year_target} years of data for {epochs_target} epochs.')


if __name__ == '__main__':
    run(year_target=5, epochs_target=500)
