import math
import numpy as np
from skimage.io import imread, imshow

def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    res = x[0][0]
    for i in range(min(len(x), len(x[0]))):
        if(x[i][i] != 0):
            res *= x[i][i]

    return res

# prod_non_zero_diag(np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]))

def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x = sorted(x)
    y = sorted(y)
    flag = len(y) == len(x)
    sz = min(len(y), len(x))
    i = 0
    for i in range(sz):
        flag = flag & (x[i] == y[i])

    return flag

# are_multisets_equal(np.array([1, 1, 2, 5, 8]), np.array([4, 2, 1, 2]))

def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    sz = len(x)
    i = 0
    mx = -1e10
    for i in range(sz - 1):
        if(x[i] == 0):
            mx = max(mx, x[i + 1])

    return mx
        

# max_after_zero(np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]))


img = imread('img.png') 
cfs = np.array([0.299, 0.587, 0.114])

def convert_image(img, coefs):  
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    
    img_gray = [[0 for w in range(len(img[0]))] for h in range(len(img))]
    for w in range(len(img)):
        for h in range(len(img[0])):
            img_gray[w][h] = 1 / 3 * (
                  img[w][h][0] * coefs[0] / 255 
                + img[w][h][1] * coefs[1] / 255  
                + img[w][h][2] * coefs[2] / 255)
    
    return img_gray


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    cnts = {}
    for num in x:
        cnts[num] = 0
    for num in x:
        cnts[num] = cnts[num] + 1

    return list(set(x)), list(cnts.values())

# print(run_length_encoding(np.array([2, 2, 2, 3, 3, 3, 5])))


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    result = []
    for arrx in x:
        cur = []
        for arry in y:
            res = 0
            for i in range(len(arry)):
                res = res + (arrx[i] - arry[i])**2
#             print(res)
            res = math.sqrt(res)
#             print(res)
            cur.append(res)
        result.append(list(cur))

    return list(result)