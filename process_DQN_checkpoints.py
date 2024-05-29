import zipfile
import os


def unzip_models():
    path = os.path.join(os.getcwd(), 'four_room_extensions', 'DQN_models')
    for file in os.listdir(path):
        if file.endswith('.zip'):
            zip_path = os.path.join(path, file)

            file_name = file.split('.')[0]
            os.makedirs(os.path.join(path, file_name), exist_ok=True)

            unzip_path = os.path.join(path, file_name)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)


def list_files():
    path = os.path.join(os.getcwd(), 'four_room_extensions', 'DQN_models')
    for folder in os.listdir(path):
        fodler_path = os.path.join(path, folder)
        if os.path.isdir(fodler_path):
            print([file for file in os.listdir(fodler_path)])
            break
        break


if __name__ == '__main__':
    # unzip_models()
    list_files()
