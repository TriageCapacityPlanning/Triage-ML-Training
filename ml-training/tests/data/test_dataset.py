from triage_ml.data.dataset import DataSet
import pytest
from datetime import datetime


@pytest.fixture(scope="module")
def test_data():
    data = []
    with open('tests/test_data.txt') as f:
        for line in f.readlines():
            line = line.rstrip().split(',')
            data.append((
                int(line[0]),
                int(line[1]),
                datetime.strptime(line[2], '%Y-%m-%d'),
            ))

    return data


def test_dataset_len():
    data = 5*[('a', 'b', 'c')]
    dataset = DataSet(data)

    assert len(dataset) == 5


def test_dataset_index():
    data = [('a', 'b', 'c'), (1, 2, 3)]
    dataset = DataSet(data)

    assert dataset[0][0] == 'a'
    assert dataset[1][0] == 1


def test_order_by():
    data = [(i, 1, 2) for i in range(5)]
    dataset = DataSet(data)

    dataset.order_by('clinic_id', descending=True)
    for i in range(5):
        assert dataset[i][0] == 4 - i


def test_aggregate_on():
    data = [(1, 2, i) for i in range(5)]
    dataset = DataSet(data)
    dataset_data = dataset.data

    agg = dataset.aggregate_on('clinic_id', key=lambda x: x)

    assert agg == {1: dataset_data}


def test_aggregate_on_dates(test_data):
    dataset = DataSet(test_data)

    agg = dataset.aggregate_on('date_received', key=lambda d: str(d.date()))

    assert len(agg['2020-01-01']) == 1
    assert len(agg['2020-01-03']) == 2
    assert len(agg['2020-01-04']) == 1
    assert len(agg['2020-01-05']) == 1
    assert len(agg['2020-01-07']) == 4
    assert len(agg['2020-01-11']) == 2
