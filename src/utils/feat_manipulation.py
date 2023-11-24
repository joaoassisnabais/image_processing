import numpy as np
from sklearn.neighbors import NearestNeighbors

def feat_matching(feats1, feats2, k=2):
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
    # NOTE: Transpose because it should be kps x (64/128) and not (64/128) x kps
    # NOTE: Same for (x,y) coordinates
    kp1, kp2 = feats1[:2].T, feats2[:2].T
    des1, des2 = feats1[2:], feats2[2:]
    
    # TODO: see if there's a better way to do that
    #Limiting the number of descriptors to a restricted number
    m = min(des1.shape[1], des2.shape[1])
    des1 = des1[:, :m].T
    des2 = des2[:, :m].T
    
    # Calculate distances between descriptors
    # Fit a k-NN model on descriptors of set 2
    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(des2)

    # Query nearest neighbors for descriptors from set 1
    distances, indices = nbrs.kneighbors(des1, n_neighbors = k)
    
    # Apply Lowe's ratio test to find good matches
    threshold = 0.75
    good_matches = []
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        if dists[0] < threshold * dists[1]:
            good_matches.append((i, idxs[0]))
    
    
    src_kps = kp1[[x for x, y in good_matches]]
    dest_kps = kp2[[y for x, y in good_matches]]
    
    return src_kps, dest_kps

