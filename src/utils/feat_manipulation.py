import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def match_feats(feats1, feats2, k=2):
    """
    Match features between two images using nearest neighbors.

    Parameters
    ----------
    feats1 : numpy.ndarray
        Features from the first image.
    feats2 : numpy.ndarray
        Features from the second image.
    k : int
        Number of nearest neighbors to use.

    Returns
    -------
    matches : numpy.ndarray
        Array of shape (N, 2) where each row is a match between two features.
        The first column is the index of the feature from the first image,
        while the second column is the index of the feature from the second
        image.
    """
    # Extract keypoints and descriptors from features
    kp1, kp2 = feats1[:2], feats2[:2]
    des1, des2 = feats1[2:], feats2[2:]
      
    # Fit a k-NN model on descriptors of set 2
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(des2)
    
    # Query nearest neighbors for descriptors from set 1
    distances, indices = nbrs.kneighbors(des1, n_neighbors=k)
    
    # Apply Lowe's ratio test to select good matches
    threshold = 0.75
    good_matches = []
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        if dists[0] < threshold * dists[1]:
            good_matches.append((i, idxs[0]))
    
    # Create matches array
    matches = np.zeros((des1.shape[0], 2))
    matches[:, 0] = np.arange(des1.shape[0])
    matches[:, 1] = indices[:, 0]
    
    return matches.astype(int)

