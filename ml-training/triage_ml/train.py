from triage_ml import train_radius_variance
from triage_ml.data.dataset import DataSet
from triage_ml.triage_api import TriageAPI
from triage_ml.data.visualizations import visualize_training_results

from typing import Text
from datetime import datetime
import os
import argparse
import requests


TRIAGE_API_URL = os.getenv('TRIAGE_API_URL')
TRIAGE_API_USER = os.getenv('TRIAGE_API_USER')
TRIAGE_API_PASS = os.getenv('TRIAGE_API_PASS')

DATE_FORMAT = '%Y-%m-%d'

MODELS = {
    'radius_variance': train_radius_variance
}


def _load_dataset_from_file(file_name: Text) -> DataSet:
    data = []
    with open(file_name) as f:
        for line in f.readlines():
            line = line.rstrip().split(',')
            data.append((
                int(line[0]),
                int(line[1]),
                datetime.strptime(line[2], '%Y-%m-%d'),
            ))

    return DataSet(data)


def parse_args():
    """
    Parser configuration
    :return: parsed augments object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', required=True, choices=MODELS.keys(),
                        help='The name of the model to train.')
    parser.add_argument('-c', '--clinic_id', required=True, type=int,
                        help='The ID of the clinic whose data to use for training.')
    parser.add_argument('-s', '--severity', required=True, type=int,
                        help='The triage severity level to train on.')
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='Number of passes through the dataset to train for.')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='The gradient descent learning rate.'),
    parser.add_argument('-p', '--persist', default=False, type=bool,
                        help='Whether training weights should be persisted to database.')

    # If pulling data from API
    parser.add_argument('-sd', '--start_date', help=f'The start date of the data. Format: {DATE_FORMAT}')
    parser.add_argument('-ed', '--end_date', help=f'The end date of the data. Format: {DATE_FORMAT}')

    # If using local data
    parser.add_argument('-d', '--dataset',
                        help='An optional local dataset to train on instead')

    return parser.parse_args()


def main(http=requests):
    """
    Entrypoint for triage-train.
    """
    args = parse_args()
    triage_api = TriageAPI(TRIAGE_API_URL, TRIAGE_API_USER, TRIAGE_API_PASS, http)

    if args.dataset:
        dataset = _load_dataset_from_file(args.dataset)
    else:
        start_date = datetime.strptime(args.start_date, DATE_FORMAT)
        end_date = datetime.strptime(args.end_date, DATE_FORMAT)
        dataset = triage_api.get_data(args.clinic_id, args.severity, start_date, end_date)

    dataset.filter_on('clinic_id', lambda c_id: c_id == args.clinic_id)
    dataset.filter_on('severity', lambda s: s == args.severity)
    trained_model, train_data, test_data, history = train_radius_variance(dataset,
                                                                          epochs=args.epochs, lr=args.learning_rate)

    trained_model.get_model().save('weights.h5')
    if args.persist:
        triage_api.post_weights(args.clinic_id, args.severity, 'weights.h5', history.history['val_loss'][-1])

    visualize_training_results(trained_model, train_data, test_data, 'results.png')
