from json import JSONEncoder, JSONDecoder
import numpy as np

class DataEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, set):
            return list(o)
        else:
            return o.__dict__

class Object2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = 2.14

class Object1:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = 3

def from_json(json_object):
    print(json_object)
    if 'fname' in json_object:
        return Object1(json_object['fname'])


K = Object1(a=Object2(x=[1, 2], y={'l': [1, 3], 'm':[2, 4], 'n': {"nin", "xis"}}), b=np.ones((2,2)))
encoded = DataEncoder().encode(K)
print(encoded)

f = JSONDecoder(object_pairs_hook=from_json).decode(encoded) # '{"fname": "/foo/bar"}'