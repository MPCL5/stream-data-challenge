import urllib.request


def download(address, save_path) -> bool:
    try:
        print(f'Downloading {address}')
        urllib.request.urlretrieve(address, save_path)
        print(f'Download complete!')
        return True
    except:
        print(f'An error occured while downloading {address}')
        return False
