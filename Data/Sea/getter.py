from skmultiflow.data import FileStream
from skmultiflow.data.base_stream import Stream
from skmultiflow.data.sea_generator import SEAGenerator
import numpy as np
import os

from Utils.generator_saver import GeneratorSaver

DATA_PATH = './Data/Sea/data.csv'
DATA_NAME = 'SEA Dataset'


class SEAGeneratorSaver(SEAGenerator, GeneratorSaver):
    def __init__(self, classification_function=0, random_state=None, balance_classes=False,
                 noise_percentage=0.0, data_path='./data'):
        SEAGenerator.__init__(self, classification_function, random_state, balance_classes, noise_percentage)
        GeneratorSaver.__init__(self, data_path=data_path)

    def next_sample(self, batch_size=1):
        X, y = super().next_sample(batch_size)
        X = np.hstack((X, X, X))  # because we have issues in low dimensionality
        self.save_data(X, y)

        return X, y


# NOTE: these functions are duplicated among project for the sake of readability.
def get_sea_stream() -> Stream:
    print(f'Getting {DATA_NAME}...')
    print(f'Loading local data from {DATA_PATH}')
    if os.path.exists(DATA_PATH):
        stream = FileStream(DATA_PATH)
        stream.basename = DATA_NAME

        if stream == 10_000:
            return stream

        os.remove(DATA_PATH)

    print('Local dataset is not fine! Generating...')
    stream = SEAGeneratorSaver(data_path=DATA_PATH)
    stream.basename = DATA_NAME
    return stream
