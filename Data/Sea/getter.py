from skmultiflow.data import FileStream
from skmultiflow.data.base_stream import Stream
from skmultiflow.data.sea_generator import SEAGenerator
from csv import writer
import numpy as np
import os

DATA_PATH = './Data/Sea/data.csv'


class SEAGeneratorSaver(SEAGenerator):
    def __init__(self, classification_function=0, random_state=None, balance_classes=False,
                 noise_percentage=0.0, data_path='./data'):
        super().__init__(classification_function, random_state, balance_classes, noise_percentage)
        self.temp_data = None
        self.data_path = data_path

    def flush_temp(self):
        """ flush_temp
        Writes the temp data to a CSV file
        """
        with open(self.data_path, 'a', newline='') as stream_file:
            writer_object = writer(stream_file)

            writer_object.writerows(self.temp_data.tolist())

            # close file
            stream_file.close()

        self.temp_data = None  # flush memory

    def save_data(self, X, y):
        """ save_data
        This function save the data in chunks and avoid memory leak
        ---

        :param X:
        :param y:
        """
        data_to_append = np.hstack((X, np.array([y]).T))
        if self.temp_data is None:
            self.temp_data = data_to_append
        else:
            self.temp_data = np.vstack((self.temp_data, data_to_append))

        if len(self.temp_data) > 500:
            self.flush_temp()

    def next_sample(self, batch_size=1):
        X, y = super().next_sample(batch_size)
        self.save_data(X, y)

        return X, y

    def __del__(self):
        self.flush_temp()

def get_sea_stream() -> Stream:
    if os.path.exists(DATA_PATH):
        stream = FileStream(DATA_PATH)
        stream.basename = 'SEA Dataset'

        if stream == 10_000:
            return stream

        os.remove(DATA_PATH)

    stream = SEAGeneratorSaver(data_path=DATA_PATH)
    stream.basename = 'SEA Dataset'
    return stream
