from skmultiflow.data import FileStream
from skmultiflow.data.base_stream import Stream
import os

from Utils.downloader import download

DATA_PATH = './Data/Spam/data.csv'
DATA_URL = 'https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow/blob/master/real-world/spam.csv?raw=true'


def spam_stream_getter() -> Stream:
    if not os.path.exists(DATA_PATH):
        download(DATA_URL, DATA_PATH)

    return FileStream(DATA_PATH)

