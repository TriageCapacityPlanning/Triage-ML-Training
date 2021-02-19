from triage_ml.data.dataset import DataSet, MLDataSet
from triage_ml.models.prediction_model import PredictionModel

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def _date_range(min_date: datetime, max_date: datetime):
    for n in range(int((max_date - min_date).days) + 1):
        yield min_date + timedelta(n)


def visualize_dataset_arrivals(dataset: DataSet, output_file: str):
    dataset.order_by('date_received')
    first = dataset[0].date_received
    last = dataset[-1].date_received
    agg = dataset.aggregate_on('date_received', lambda dr: str(dr.date()))
    x = list(_date_range(first, last))
    x2 = []
    y = []
    i = 0
    for x_val in x:
        i += 1
        x2.append(i)
        y.append(len(agg[str(x_val.date())]) if str(x_val.date()) in agg else 0)

    plt.plot_date(x, y, markersize=2)
    plt.gcf().autofmt_xdate()
    plt.savefig(output_file)


def visualize_training_results(model: PredictionModel, train_data: MLDataSet, test_data: MLDataSet, output_file: str):
    train_size = len(train_data.inputs[0])
    test_size = len(test_data.inputs[0])

    train = []
    gt = []
    pred = []

    for i in range(train_size):
        train.append(train_data.outputs[0][i][0])
        gt.append(None)
        pred.append(None)

    # data = [x[np.newaxis, -1] for x in train_data.inputs]
    # print(data)
    # print(model.get_model()(data))
    preds = model.predict([x[np.newaxis, -1] for x in train_data.inputs], test_data.inputs[1])
    for i in range(test_size):
        gt.append(test_data.outputs[0][i][0])
        pred.append(preds[i][0][0])

    plt.plot(train, label='Train data')
    plt.plot(gt, label='Test data')
    plt.plot(pred, label='Predicted')
    plt.legend()
    plt.savefig(output_file)

