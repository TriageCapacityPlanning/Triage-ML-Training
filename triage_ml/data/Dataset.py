from __future__ import annotations

from collections import namedtuple
from typing import Any, List, Tuple, Text, Callable

_ATTRS = ['id', 'clinic_id', 'severity', 'date_received', 'date_seen']

DataPoint = namedtuple('DataPoint', _ATTRS)


class DataSet:

    def __init__(self, data: List[Tuple]):
        # Tuple must fit structure of DataPoint
        if not len(data) or len(data[0]) == len(_ATTRS):
            raise ValueError('Invalid data argument.')

        self.data = [DataPoint(*item) for item in data]

    def filter_on(self, attribute: Text, predicate: Callable[[Any], bool]) -> DataSet:
        self.data = filter(lambda data_point: predicate(data_point[_ATTRS.index(attribute)]), self.data)
        return self
