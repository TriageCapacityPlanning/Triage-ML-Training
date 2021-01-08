class PredictionModel:

    def create_dataset(self, data):
        raise NotImplementedError()

    def save_dataset(self, dataset):
        raise NotImplementedError()

    def load_dataset(self, dataset_file):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    def preprocess_input(self, model_input):
        return model_input

    def postprocess_output(self, model_output):
        return model_output
