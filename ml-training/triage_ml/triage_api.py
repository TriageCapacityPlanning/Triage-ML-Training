import requests
from datetime import datetime


class TriageAPI:

    DATE_FORMAT = '%y-%M-%d'

    def __init__(self, url, username, password, http=requests):
        self.url = url
        self.username = username
        self.password = password
        self.http = http

    def get_data(self, clinic_id: int, severity: int, start_date: datetime, end_date: datetime):
        data = {
            'interval': (start_date.strftime(self.DATE_FORMAT), end_date.strftime(self.DATE_FORMAT))
        }

        res = self.http.post(f'{self.url}/data/{clinic_id}/{severity}', json=data)
        res.raise_for_status()

    def post_weights(self, clinic_id: int, severity: int, weights_path: str, accuracy: float, make_in_use: bool = False):
        with open(weights_path, 'rb') as weights_file:
            data = {
                'model_weights': weights_file,
                'clinic_id': (None, clinic_id),
                'severity': (None, severity),
                'accuracy': (None, str(accuracy)),
                'make_in_use': (None, make_in_use),
            }
            res = self.http.post(f'{self.url}/upload/model', files=data)
            res.raise_for_status()
