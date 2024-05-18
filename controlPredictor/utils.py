import json
import pickle

def read_file(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def read_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def write_to_pickle(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
