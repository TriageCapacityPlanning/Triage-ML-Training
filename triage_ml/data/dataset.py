from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict, List, Tuple, Text, Callable
import numpy as np

_ATTRS = ['id', 'clinic_id', 'severity', 'date_received', 'date_seen']

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
        if not len(data) or len(data[0]) != len(_ATTRS):
            raise ValueError('Invalid data argument.')

        self.data = [DataPoint(*item) for item in data]

    def filter_on(self, attribute: Text, predicate: Callable[[Any], bool]) -> DataSet:
        """
        Filter the DataSet on a certain attribute with the given predicate.

        Any data points whose specified attribute does not meet the predicate are
        removed.
        :param attribute: The attribute to filter on.
        :param predicate: The predicate function
        :return: The modified DataSet
        """
        attr_idx = _ATTRS.index(attribute)
        self.data = filter(lambda data_point: predicate(data_point[attr_idx]), self.data)
        return self

    def order_by(self, attribute: Text, descending=False) -> DataSet:
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MLDataSet:

    def __init__(self, inputs: List[np.array], outputs: List[np.array]):
        self.inputs = inputs
        self.outputs = outputs

    def split(self, point=0.5) -> Tuple[MLDataSet, MLDataSet]:
        """
        Split the MLDataSet into two MLDataSets at a given point
        :param point: The point to split the MLDataSet (0,1)
        :return: A Tuple (MLDataSet, MLDataSet)
        """
        split_idx = len(self.inputs[0]) * point
        s1 = MLDataSet([x[:split_idx] for x in self.inputs],
                       [y[:split_idx] for y in self.outputs])
        s2 = MLDataSet([x[split_idx:] for x in self.inputs],
                       [y[split_idx:] for y in self.outputs])
        return s1, s2
