import os
import pickle


def frames_load(dir_name):
    frames = {}
    for file_name in os.listdir(dir_name):
        if '.' in file_name and file_name.split('.')[1] == 'data':
            with open('{}/{}'.format(dir_name, file_name), 'rb') as file:
                data = pickle.load(file)
                frames.update(data)
    return frames
