import numpy as np
from scipy.spatial.distance import cdist

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
    kp1, kp2 = feats1[:2], feats2[:2]
    des1, des2 = feats1[2:], feats2[2:]
    
    # TODO: see if there's a better way to do that
    #Limiting the number of descriptors to a restricted number
    # NOTE: Transpose because it should be 2000x64 given that there are 2000 kps with 64 descriptors
    m = max(des1.shape[1], des2.shape[1])
    des1 = des1[:, :m].T
    des2 = des2[:, :m].T
    
    # Calculate distances between descriptors
    # NOTE: Using cdist because it's supports cosine distance
    distances = cdist(des1, des2, metric='cosine')

    # Apply ratio test to find good matches
    threshold = 0.75
    good_matches = []
    for i, dist in enumerate(distances):
        sorted_indices = np.argsort(dist)
        best_match_idx = sorted_indices[0]
        second_best_match_idx = sorted_indices[1]

        if dist[best_match_idx] < threshold * dist[second_best_match_idx]:
            good_matches.append((i, best_match_idx))
    


    return kp1, kp2

