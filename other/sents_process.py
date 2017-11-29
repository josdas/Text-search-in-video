import os
import pickle
import re


def sent2words(sent):
    return [word
            for word in re.sub(r'[^a-z ]+', '', str.lower(sent)).split(' ')
            if word not in ['', ' ', '<']]


def sents_load(dir_name):
    texts = {}
    for file_name in os.listdir(dir_name + '/w2v'):
        if '.' in file_name and file_name.split('.')[1] == 'wtov':
            with open('{}/w2v/{}'.format(dir_name, file_name), 'rb') as fl:
                data = pickle.load(fl)
                for id, v in data:
                    texts[id] = texts.get(id, []) + [v]
    return texts
