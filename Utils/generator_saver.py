from abc import ABC
from csv import writer
import numpy as np
import io


class GeneratorSaver(ABC):
    def __init__(self, data_path='./data.csv'):
        self.temp_data = None
        self.data_path = data_path

    def flush_temp(self):
        """ flush_temp
        Writes the temp data to a CSV file
        """
        with io.open(self.data_path, 'a', newline='') as stream_file:
            writer_object = writer(stream_file)

            writer_object.writerows(list(self.temp_data))

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

    def __del__(self):
        if self.temp_data is not None:
            self.flush_temp()
