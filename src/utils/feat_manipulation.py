import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors

from utils.homography import homography

def feat_matching(feats1, feats2, k=30):
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
    threshold = 0.85
    good_matches = []
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        if dists[0] < threshold * dists[1]:
            good_matches.append((i, idxs[0]))
    
    src_kps = kp1[[i for i, _ in good_matches]]
    dest_kps = kp2[[i for _, i in good_matches]]
    match1to2 = [[np.where(kp1==src_kps[i])[0][0], np.where(kp2==dest_kps[i])[0][0]] for i in range(len(good_matches))]
    
    return src_kps, dest_kps, match1to2

def RANSAC(src_points, dest_points, threshold = 1):
    """
    Conducts RANSAC analysis on two sets of matching keypoints (source and destination).
    It iteratively seeks the model that best represents the entire dataset by 
    identifying the model that generates the highest count of inliers.
    
    Parameters
    ----------
    src_points : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the first image that have a corresponding point in the second image.
    dest_points : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the second image that have a corresponding point in the first image.
    threshold : float
        Distance threshold to consider a point as an inlier.
        
    Returns
    -------
    best_src_inliers : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the first image that have a corresponding point in the second image.
    best_dest_inliers : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the second image that have a corresponding point in the first image.
    best_inlier_mask : numpy.ndarray
        Array of shape (N, ) where we have a boolean mask of the inliers.
    """
    
    # TODO: what is a good error threshold?

    P = 0.99    # Prob of success
    n = 4       # Number of samples
    p = 0.5     # Prob of choosing an inlier (safe bet)
    num_iterations = math.ceil(math.log(1 - P) / math.log(1 - (p ** n)))  # Iterations to be run in order to reach success P
    
    best_src_inliers = None
    best_dest_inliers = None
    best_n_inliers = np.NINF
    best_inlier_mask = None
    
    for _ in range(num_iterations):
        # Randomly select some points to form a line model
        ids = np.random.choice(len(src_points), 6, replace = False)
        
        src_inliers = src_points[ids, :]
        dest_inliers = dest_points[ids, :]
        
        # Compute the homography
        H = homography(src_inliers, dest_inliers)
        
        # Apply the transformation to the source points
        src_img = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
        res = H @ src_img.T
        res = (res[:2, :] / res[2, :]).T

        # Compute the distance between the transformed points and the destination points
        inlier_mask = np.linalg.norm(res - dest_points, axis = 1) < threshold
        n_inliers = inlier_mask.sum()

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_src_inliers = src_points[inlier_mask, :]
            best_dest_inliers = dest_points[inlier_mask, :]
            best_inlier_mask = inlier_mask
            
    return best_src_inliers, best_dest_inliers, best_inlier_mask