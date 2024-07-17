import pickle

def read_pkl(path):
    with open(path, 'rb') as fr:
        data = pickle.load(fr)
    return data

def write_pkl(path, data):
    with open(path, 'wb') as fout:
        pickle.dump(data, fout)