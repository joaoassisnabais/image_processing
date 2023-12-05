import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat

from utils.read import parse_config_file
from utils.debug_funcs import compare_process_video_sift

debug = False

def get_features(img, sift):
    """
    Compute the features of an image using SIFT
    
    Parameters
    ----------
    img : numpy.ndarray
        The image to compute the features for
    sift : cv2.SIFT
        The SIFT detector to use
        
    Returns
    -------
    features : numpy.ndarray
        The features of the image
    """
    # Detect keypoints and compute their descriptors using SIFT
    keypoints, descriptors = sift.detectAndCompute(img, None)
    features = []
    for i in range(len(keypoints)):
        features += [[keypoints[i].pt[0], keypoints[i].pt[1]] + np.ndarray.tolist(descriptors[i])]
        
    return np.asarray(features).T

def main(config_file):
    
    # Read the config file
    if config_file is not None:
        config = parse_config_file(config_file)
        map_path = config['image_map'][0][0]
        out_path = config['keypoints_out'][0][0]
        
    out_path = os.path.join(os.path.dirname(__file__), 'out', 'features.mat')
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create(2000)
    # Open the video file
    cap = cv2.VideoCapture('src/data/backcamera_s1.mp4')

    all_features = np.empty((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),),dtype=object)
    frame_count = 0

    #If there is a map image, compute the features for it
    if map_path:
        map_img = cv2.imread(map_path)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        all_features[frame_count] = get_features(map_img, sift)
        frame_count += 1

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            all_features[frame_count] = get_features(frame, sift)
            frame_count += 1
        else:
            cap.release()
            break
    
    all_features = np.asarray(all_features, dtype=object)

    out_mat_format = {'features': all_features}
    savemat(out_path, out_mat_format)

    if debug:
        compare_process_video_sift("src/data/backcamera_s1.mp4", out_path, 0)

if __name__ == '__main__':
    main()

