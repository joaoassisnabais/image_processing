import cv2
import numpy as np
from scipy.io import savemat

# Initialize SIFT detector
sift = cv2.SIFT_create(10)

# Open the video file
cap = cv2.VideoCapture('src/data/backcamera_s1.mp4')

all_keypoints = []
all_descriptors = []

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Detect keypoints and compute their descriptors using SIFT
        keypoints, descriptors = sift.detectAndCompute(frame, None)

        # Convert keypoints to a numpy array of tuples (x, y) and store
        keypoints_array = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints], dtype=np.float32)
        all_keypoints.append(keypoints_array)

        # Convert descriptors to a numpy array and store
        if descriptors is not None:
            all_descriptors.append(descriptors)
        else:
            # Append an empty array if no descriptors are found
            all_descriptors.append(np.array([], dtype=np.float32))

    else:
        break

cap.release()

# Convert lists of arrays to a format compatible with savemat
keypoints_struct = {'frame_{}'.format(i): k for i, k in enumerate(all_keypoints)}
descriptors_struct = {'frame_{}'.format(i): d for i, d in enumerate(all_descriptors)}

# Save the structured data to a .mat file
savemat('src/data/keypoints_and_descriptors.mat', {'keypoints': keypoints_struct, 'descriptors': descriptors_struct})




