import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat

from utils.debug_funcs import compare_process_video_sift

# TODO: understand what the each point in the config file means
def main():
    # Initialize SIFT detector
    sift = cv2.SIFT_create(2000)
    # Open the video file
    cap = cv2.VideoCapture('src/data/backcamera_s1.mp4')

    all_features = np.empty((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),),dtype=object)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Detect keypoints and compute their descriptors using SIFT
            keypoints, descriptors = sift.detectAndCompute(frame, None)

            features = []
            for i in range(len(keypoints)):
                features += [[keypoints[i].pt[0], keypoints[i].pt[1]] + np.ndarray.tolist(descriptors[i])]
            all_features[frame_count] = np.asarray(features).T

            frame_count += 1
        else:
            break
    cap.release()
    
    all_features = np.asarray(all_features, dtype=object)

    out_mat_format = {'features': all_features}
    out_path = os.path.join(os.path.dirname(__file__), 'out', 'features.mat')
    savemat(out_path, out_mat_format)

    compare_process_video_sift("src/data/backcamera_s1.mp4", out_path, 0)

if __name__ == '__main__':
    main()

