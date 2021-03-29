from triage_ml.models.radius_variance import RadiusVariance
from triage_ml.train import main

import uuid
import os


class FakeObj:

    def __getitem__(self, key):
        return 'test'


class FakeResponse:

    def json(self):
        return FakeObj()

    def raise_for_status(self):
        return


class FakeRequests:

    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        return FakeResponse()


def test_load_data_from_database():
    """SRS: MOD-1"""
    args = ['-m=radius_variance', '-c=0', '-s=1', '-sd=2010-01-01', '-ed=2020-01-01']

    requests = FakeRequests()
    try:
        main(http=requests, str_args=args)
    except AttributeError as ex:
        assert str(ex) == "'NoneType' object has no attribute 'filter_on'"

    requests.calls.index(('None/data/0/1', {'interval': ('10-00-01', '20-00-01')}, {'Authorization': 'Bearer test'}))


def test_write_to_disk():
    """SRS: MOD-3"""
    model = RadiusVariance()
    file_name = str(uuid.uuid4()) + '.h5'
    model.get_model().save(file_name)

    assert os.path.isfile(file_name)
    os.remove(file_name)
    assert not os.path.isfile(file_name)
