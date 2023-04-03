from skmultiflow.data import FileStream
from skmultiflow.data.base_stream import Stream
import os

from Utils.downloader import download

DATA_PATH = './Data/Spam/data.csv'
DATA_URL = 'https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow/blob/master/real-world/spam.csv?raw=true'


def get_spam_stream() -> Stream:
    if not os.path.exists(DATA_PATH):
        print('Spam dataset does not exists on local!')
        download(DATA_URL, DATA_PATH)

    stream = FileStream(DATA_PATH)
    stream.basename = 'Spam'  # customize stream name

    return stream

