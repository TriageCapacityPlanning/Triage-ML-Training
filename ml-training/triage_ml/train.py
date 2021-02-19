from triage_ml import train_radius_variance
from triage_ml.data.dataset import DataSet
from triage_ml.triage_api import TriageAPI
from triage_ml.data.visualizations import visualize_training_results

from typing import Text
from datetime import datetime
import os
import argparse


TRIAGE_API_URL = os.getenv('TRIAGE_API_URL')
TRIAGE_API_USER = os.getenv('TRIAGE_API_USER')
TRIAGE_API_PASS = os.getenv('TRIAGE_API_PASS')

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
                int(line[2]),
                datetime.strptime(line[3], '%Y-%m-%d'),
                datetime.strptime(line[4], '%Y-%m-%d')
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
                        help='The gradient descent learning rate.')

    # If pulling data from API
    parser.add_argument('-sd', '--start_date', help='The start date of the data.')
    parser.add_argument('-ed', '--end_date', help='The end date of the data.')

    # If using local data
    parser.add_argument('-d', '--dataset',
                        help='An optional local dataset to train on instead')

    return parser.parse_args()


def main():
    """
    Entrypoint for triage-train.
    """
    args = parse_args()
    triage_api = TriageAPI(TRIAGE_API_URL, TRIAGE_API_USER, TRIAGE_API_PASS)

    if args.dataset:
        dataset = _load_dataset_from_file(args.dataset)
    else:
        pass  # dataset = triage_api.get_dataset

    trained_model, train_data, test_data = train_radius_variance(dataset, epochs=args.epochs, lr=args.learning_rate)

    # TODO(samcymbaluk): Write results to DB

    visualize_training_results(trained_model, train_data, test_data, 'results.png')
