from triage_ml import train_radius_variance
from triage_ml.data.dataset import DataSet

from typing import Text
from datetime import datetime


def _load_dataset_from_file(file_name: Text) -> DataSet:
    data = []
    with open(file_name) as f:
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


def main():
    # TODO(samcymbaluk): Load dataset from DB
    dataset = _load_dataset_from_file('tests/test_data.txt')

    train_radius_variance(dataset, epochs=50)

    # TODO(samcymbaluk): Write results to DB
