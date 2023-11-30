import numpy as np


def homography(src, dest):
    """
    Compute the homography tranformation between two sets of points

    Parameters
    ----------
    src : numpy.ndarray
        Features from the first image.
    dest : numpy.ndarray
        Features from the second image.

    Returns
    -------
    A : numpy.ndarray
        Array of shape (3, 3) representing the transformation
    """
    src_points = src
    dest_points = dest

    n = src_points.shape[0]


    A = []
    for i in range(len(src)):
        x, y = src[i]
        u, v = dest[i]
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v])

    A = np.array(A)

    b = []
    for i in range(len(src)):
        x, y = src[i]
        u, v = dest[i]
        b.append(u)
        b.append(v)
    b = np.array(b)

    #Least squares
    H = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.concatenate((H, np.array([1])))


    return H.reshape((3, 3))


