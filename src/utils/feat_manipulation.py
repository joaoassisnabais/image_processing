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
    src_kps : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the first image that have a corresponding point in the second image.
        
    dest_kps : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the second image that have a corresponding point in the first image.

    matches1to2 : array
        Array of shape (N, 2) TO BE USED FON DEBUG ONLY. 	
        Matches from the first image to the second one, which means that keypoints1[i] has a corresponding point in keypoints2[matches1to2[i]] 

    """
    # Extract keypoints and descriptors from features
    # NOTE: Transpose because it should be kps x (64/128) and not (64/128) x kps
    # NOTE: Same for (x,y) coordinates
    kp1, kp2 = feats1[:2].T, feats2[:2].T
    des1, des2 = feats1[2:], feats2[2:]
    
    # TODO: see if there's a better way to do that
    #Limiting the descriptors to a restricted number
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
    matches1to2 = [[np.where(kp1==src_kps[i])[0][0], np.where(kp2==dest_kps[i])[0][0]] for i in range(len(good_matches))]
    
    return src_kps, dest_kps, matches1to2

