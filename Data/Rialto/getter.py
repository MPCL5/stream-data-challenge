from skmultiflow.data import FileStream
from skmultiflow.data.base_stream import Stream
import os

from Utils.downloader import download

DATA_PATH = './Data/Rialto/data.csv'
DATA_URL = 'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/rialto.csv'
DATA_NAME = 'Rialto'


def get_rialto_stream() -> Stream:
    if not os.path.exists(DATA_PATH):
        print('Rialto dataset does not exists on local!')
        download(DATA_URL, DATA_PATH)

    stream = FileStream(DATA_PATH)
    stream.basename = DATA_NAME  # customize stream name

    return stream
