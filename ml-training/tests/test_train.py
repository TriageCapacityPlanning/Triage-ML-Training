from triage_ml.models.radius_variance import RadiusVariance

import uuid
import os


def test_load_data_from_database():
    # Not yet implemented
    assert False


def test_validation():
    # Not yet implemented
    assert False


def test_write_to_disk():
    model = RadiusVariance()
    file_name = str(uuid.uuid4()) + '.h5'
    model.get_model().save(file_name)

    assert os.path.isfile(file_name)
    os.remove(file_name)
    assert not os.path.isfile(file_name)
