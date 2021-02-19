from collections import namedtuple
from typing import Any, Dict, List, Tuple, Text, Callable
import numpy as np

_ATTRS = ['clinic_id', 'severity', 'date_received']

DataPoint = namedtuple('DataPoint', _ATTRS)


class DataSet:
    """
    Stores a set of patient arrival data pulled from the Triage Database.

    Attributes:
        data: A list of DataPoint tuples.
    """

    def __init__(self, data: List[Tuple]):
        """
        Create a new DataSet from a list of tuples.

        Tuples must fit the structure (length) of DataPoint.
        :param data: The list of data tuples.
        """
        # Tuple must fit structure of DataPoint
        if len(data) and len(data[0]) != len(_ATTRS):
            raise ValueError('Invalid data argument.')

        self.data = [DataPoint(*item) for item in data]

    def filter_on(self, attribute: Text, predicate: Callable[[Any], bool]):
        """
        Filter the DataSet on a certain attribute with the given predicate.

        Any data points whose specified attribute does not meet the predicate are
        removed.
        :param attribute: The attribute to filter on.
        :param predicate: The predicate function
        :return: The modified DataSet
        """
        attr_idx = _ATTRS.index(attribute)
        self.data = list(filter(lambda data_point: predicate(data_point[attr_idx]), self.data))
        return self

    def order_by(self, attribute: Text, descending=False):
        """
        Orders the list of DataPoints by a given attribute.
        :param attribute: The attribute to order by.
        :param descending: A boolean, whether the data should ordered as descending.
        :return: The modified DataSet
        """
        attr_idx = _ATTRS.index(attribute)
        self.data = sorted(self.data, key=lambda data_point: data_point[attr_idx], reverse=descending)
        return self

    def aggregate_on(self, attribute: Text, key: Callable[[Any], Any]) -> Dict[Any, List[DataPoint]]:
        """
        Produces an aggregation of DataPoints where the attribute keys are equal.
        :param attribute: The attribute to aggregate on.
        :param key: A function to get the equivalence property of the attribute.
        :return: A Dictionary of attributes to their aggregated DataPoints.
        """
        attr_idx = _ATTRS.index(attribute)
        sorted_data = sorted(self.data, key=lambda data_point: data_point[attr_idx])

        aggregation = {}
        agg = None
        for data_point in sorted_data:
            if agg:
                if key(data_point[attr_idx]) == key(agg[1][-1][attr_idx]):
                    agg[1].append(data_point)
                    continue
                else:
                    aggregation[key(agg[0])] = agg[1]

            agg = (data_point[attr_idx], [data_point])

        if agg:
            aggregation[key(agg[0])] = agg[1]

        return aggregation

    def write_to_file(self, file_name: str):
        """
        Writes the DataSet to the provided file.
        :param file_name: The file to write to.
        """
        lines = []
        for datum in self.data:
            serialized_datum = (
                str(datum.clinic_id),
                str(datum.severity),
                datum.date_received.strftime('%Y-%m-%d'),
            )
            lines.append(','.join(serialized_datum))

        with open(file_name, 'w') as file:
            file.write('\n'.join(lines))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MLDataSet:

    def __init__(self, seq_inputs: List[np.array], other_inputs: List[np.array], outputs: List[np.array], seq_size):
        self.seq_size = seq_size
        self.inputs = [self._setup_seq(x) for x in seq_inputs] + [x[seq_size:] for x in other_inputs]
        self.outputs = [y[seq_size:] for y in outputs]

    def _setup_seq(self, data):
        seq_data = []
        for i in range(self.seq_size, len(data)):
            seq_data.append(data[i-self.seq_size:i])
        seq_data = np.stack(seq_data) if seq_data else np.array(seq_data)
        return seq_data

    def split(self, point=0.5):
        """
        Split the MLDataSet into two MLDataSets at a given point
        :param point: The point to split the MLDataSet (0,1)
        :return: A Tuple (MLDataSet, MLDataSet)
        """
        split_idx = int(len(self.inputs[0]) * point)
        s1 = MLDataSet([], [x[:split_idx] for x in self.inputs],
                       [y[:split_idx] for y in self.outputs], self.seq_size)
        s2 = MLDataSet([], [x[split_idx:] for x in self.inputs],
                       [y[split_idx:] for y in self.outputs], self.seq_size)
        return s1, s2
