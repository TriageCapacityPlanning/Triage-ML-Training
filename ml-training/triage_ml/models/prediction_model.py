from triage_ml.data.dataset import DataSet, MLDataSet
from tensorflow.keras.models import Model


class PredictionModel:

    def create_ml_dataset(self, dataset: DataSet) -> MLDataSet:
        raise NotImplementedError()

    def get_model(self) -> Model:
        raise NotImplementedError()

    def predict(self, seed_data, length):
        raise NotImplementedError()

    def preprocess_input(self, model_input):
        return model_input

    def postprocess_output(self, model_output):
        return model_output
