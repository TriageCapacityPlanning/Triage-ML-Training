import argparse
import math
import random
from datetime import datetime, timedelta

from triage_ml.data.dataset import DataSet
import triage_ml.data.visualizations as visualizations


SEVERITY_PROBS = [0.1, 0.4, 1.0]


def get_random_severity():
    ran = random.random()
    for i in range(len(SEVERITY_PROBS)):
        if ran < SEVERITY_PROBS[i]:
            return i


def cyclic(start_date: datetime, end_date: datetime, random_multiple=0) -> DataSet:
    fn = lambda d: max(round(
        20 + d/300 + 2*math.sin(2 * math.pi * d/365) + random_multiple*random.random()
    ), 0)
    data = []
    i = 0
    for d, date in enumerate(date_range(start_date, end_date)):
        arrivals = fn(d)
        for arrival in range(arrivals):
            data.append((1, get_random_severity(), date))
            i += 1

    return DataSet(data)


DATE_FORMAT = '%Y-%m-%d'
GEN_METHODS = {
    'cyclic': cyclic,
    'random_cyclic': lambda sd, ed: cyclic(sd, ed, 2)
}


def date_range(min_date: datetime, max_date: datetime):
    for n in range(int((max_date - min_date).days) + 1):
        yield min_date + timedelta(n)


def parse_args():
    """
    Parser configuration
    :return: parsed augments object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--method', required=True, choices=GEN_METHODS.keys(),
                        help='The method to use to generate data.')
    parser.add_argument('-sd', '--start_date', required=True,
                        help=f'The start date of the data. Format: {DATE_FORMAT}')
    parser.add_argument('-ed', '--end_date', required=True,
                        help=f'The end date of the data. Format: {DATE_FORMAT}')
    parser.add_argument('-o', '--output_file', default='generated_data.txt',
                        help='The file to output the data to.')
    parser.add_argument('-v', '--visualization', type=str, default=None,
                        help='If set, will write a visualization image to the specified path.')

    return parser.parse_args()


def main():
    args = parse_args()
    start_date = datetime.strptime(args.start_date, DATE_FORMAT)
    end_date = datetime.strptime(args.end_date, DATE_FORMAT)

    dataset = GEN_METHODS[args.method](start_date, end_date)

    dataset.write_to_file(args.output_file)

    if args.visualization:
        visualizations.visualize_dataset_arrivals(dataset, args.visualization)


