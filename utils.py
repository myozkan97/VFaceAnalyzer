import os
import shutil
import gdown


def get_or_download(file_name: str, link: str) -> str:
    """If file does not exists, it downloads it. If it exists or have been downloaded, it returns the file path."""
    if not os.path.isfile(".\\" + file_name):
        print("%s is missing, downloading model weights." % file_name)
        print(gdown.download(link, file_name))

    return ".\\" + file_name


def delete_contents_of_folder(path: str) -> None:
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
