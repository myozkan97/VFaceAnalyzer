import os, gdown


def get_or_download(file_name: str, link: str) -> str:
    """If file does not exists, it downloads it. If it exists or have been downloaded, it returns the file path."""
    if not os.path.isfile(".\\" + file_name):
        print("%s is missing, downloading model weights." % file_name)
        print(gdown.download(link, file_name))

    return ".\\" + file_name
