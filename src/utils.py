import numpy as np

def _argmax(vec):
    return np.random.choice(np.where(vec == vec.max())[0])

def argmax_with_random_ties(arr, axis=1):
    return np.apply_along_axis(_argmax, axis=axis, arr=arr)