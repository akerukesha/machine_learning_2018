import numpy as np
import math
from scipy.spatial.distance import cdist


def prod_non_zero_diag_vect(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """

    # SOURCE: http://www.cyberforum.ru/python/thread1344307.html
    diag = x.diagonal()
    diag_no_0 = diag[diag != 0]
    mult = np.multiply.reduce(diag_no_0)
    
    return mult

# prod_non_zero_diag(np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]))

def are_multisets_equal_vect(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    x = np.sort(x, axis=None)
    y = np.sort(y, axis=None)
    z = x == y
    return np.all(z)

# are_multisets_equal(np.array([1, 2, 2, 4]), np.array([4, 2, 1, 2]))

def max_after_zero_vect(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    # SOURCE: https://stackoverflow.com/questions/4588628/find-indices-of-elements-equal-to-zero-in-a-numpy-array
    sz = np.size(x)
#     print(sz)
    y = np.where(x == 0)[0]
    y = y + 1
    y = y[(y < sz)]
    z = x[y]
#     print(x)
#     print(y)
    if len(z) == 0:
        return -1e10
    else:
        return max(z)

# max_after_zero(np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]))

def convert_image_vect(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """

    return 1 / 3 * (img[:, :, 0] * coefs[0] / 255 + 
                    img[:, :, 1] * coefs[1] / 255 + 
                    img[:, :, 2] * coefs[2] / 255)

# x = np.array([2, 2, 2, 3, 3, 3, 5]) 
# (np.array([2, 3, 5]), np.array([3, 3, 1]))

def run_length_encoding_vect(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    return np.unique(x, return_counts=True)

# print(run_length_encoding(np.array([2, 2, 2, 3, 3, 3, 5])))


# x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# y = [[10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]]


def pairwise_distance_vect(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    

    return cdist(x, y)

# print(pairwise_distance(x, y))
