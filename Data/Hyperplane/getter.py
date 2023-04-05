from skmultiflow.data import FileStream
from skmultiflow.data.base_stream import Stream
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
import os

from Utils.generator_saver import GeneratorSaver

DATA_PATH = './Data/Hyperplane/data.csv'
DATA_NAME = 'Hyperplane Dataset'


class HyperplaneGeneratorSaver(HyperplaneGenerator, GeneratorSaver):
    def __init__(self, random_state=None, n_features=10, n_drift_features=2, mag_change=0.0,
                 noise_percentage=0.05, sigma_percentage=0.1, data_path='./data.csv'):
        HyperplaneGenerator.__init__(self, random_state=random_state, n_features=n_features,
                                     n_drift_features=n_drift_features, mag_change=mag_change,
                                     noise_percentage=noise_percentage, sigma_percentage=sigma_percentage)

        GeneratorSaver.__init__(self, data_path=data_path)

    def next_sample(self, batch_size=1):
        X, y = super().next_sample(batch_size)
        self.save_data(X, y)

        return X, y


# NOTE: these functions are duplicated among project for the sake of readability.
def get_hyperplane_stream() -> Stream:
    print(f'Getting {DATA_NAME}...')
    print(f'Loading local data from {DATA_PATH}')
    if os.path.exists(DATA_PATH):
        stream = FileStream(DATA_PATH)
        stream.basename = DATA_NAME

        if stream == 10_000:
            return stream

        os.remove(DATA_PATH)

    print('Local dataset is not fine! Generating...')
    stream = HyperplaneGeneratorSaver(data_path=DATA_PATH)
    stream.basename = DATA_NAME
    return stream
