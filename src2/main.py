import cv2
import numpy as np
import os

from utils.read import parse_config_file, convert_file_path

# Function to undistort an image using the provided intrinsics and distortion coefficients
def undistort_image(image, intrinsics, distortion_coefficients):
    return cv2.undistort(image, intrinsics, distortion_coefficients)

# Function to detect feature points and match them across images
def detect_and_match_features(image1, image2):
    # Detect feature points in each image
    feature_detector = cv2.ORB_create()
    kp1, des1 = feature_detector.detectAndCompute(image1, None)
    kp2, des2 = feature_detector.detectAndCompute(image2, None)

    # Match feature points using Brute-Force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Select the best matches using the Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good.append(m)

    return kp1, kp2, good

# Function to calculate the homography matrix between two images
def compute_homography(kp1, kp2, good):
    pts1 = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return homography

# Function to estimate the camera pose based on the homography matrix
def estimate_pose(homography):
    # Extract the rotation and translation components from the homography matrix
    R, t = cv2.Rodrigues(homography[:, :2])
    translation = homography[:, 2]

    return R, translation

# Main program execution
def main():
    # Intrinsic parameters of the cameras
    conf = parse_config_file('conf_file.cfg')
    intrinsics = np.array(conf['intrinsics'], dtype=np.float32)
    distortion_coefficients = np.array(conf['distortion_coefficients'], dtype=np.float32)
    images_path = convert_file_path(conf['videos'][0][0])    
    
    # Read the images or video frames
    if isinstance(images_path, str) and images_path.endswith('.mp4'):
        # Read video frames
        cap = cv2.VideoCapture(images_path)

        # Initialize a list to store the transformations
        transformations = []

        # Process each video frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Undistort the frame if necessary
            frame = undistort_image(frame, intrinsics[0], distortion_coefficients)

            # Detect and match feature points
            kp1, kp2, good = detect_and_match_features(frame, image1)

            # Compute the homography matrix
            homography = compute_homography(kp1, kp2, good)

            # Estimate the camera pose
            R, translation = estimate_pose(homography)

            # Update the transformations list
            transformations.append((homography, R, translation))

        cap.release()

    else:
        # Read images from a folder
        for filename in os.listdir(images_path):
            if filename.endswith('.jpg'):
                # Read the image
                image = cv2.imread(os.path.join(images_path, filename))

                # Undistort the image if necessary
                image = undistort_image(image, intrinsics[int(filename.split('.')[0])], distortion_coefficients)

                # Detect and match feature points
                kp1, kp2, good = detect_and_match_features(image1, image)

                # Compute the homography matrix
                homography = compute_homography(kp1, kp2, good)

                # Estimate the camera pose
                R, translation = estimate_pose(homography)

                # Update the transformations list
                transformations.append((homography, R, translation))

    # Write the computed transformations to a file
    with open('transformations.txt', 'w') as f:
        for homography, R, translation in transformations:
            f.write('homography: {}\n'.format(homography))
            f.write('R: {}\n'.format(R))
            f.write('translation: {}\n'.format(translation))

if  __name__ == '__main__':
    main()