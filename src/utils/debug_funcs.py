import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.image as mpimg


def get_single_frame(file_path, frame_number):
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return None

    return frame

def compare_process_video_sift(video_path, out_mat_path, frame_index):
    sift = cv2.SIFT_create(2000)

    frame1 = get_single_frame(video_path, frame_index)

    f = loadmat(out_mat_path)
    feat = f['features']    
    feat = feat.squeeze() # remove the extra dimension
    feat_frame = feat[frame_index]
    print(feat_frame.shape[1])
    kp1 = feat_frame[:2].T
    print(kp1)

    keypointsCV1 = []
    for i in range(len(kp1)):
        keypointsCV1.append(cv2.KeyPoint(int(kp1[i][0]), int(kp1[i][1]), 1))

    keypoints1_sift, descriptors1 = sift.detectAndCompute(frame1, None)

    image1 = cv2.drawKeypoints(frame1, keypointsCV1, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    image2 = cv2.drawKeypoints(frame1, keypoints1_sift, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    _, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    axes[0].imshow(image1)
    axes[0].axis('off') 

    axes[1].imshow(image2)
    axes[1].axis('off') 
    axes[1].set_title("sift")

    plt.show()

def show_image_and_keypoints(video_path, index1, index2, keypoints1, keypoints2):

    frame1 = get_single_frame(video_path, index1)
    frame2 = get_single_frame(video_path, index2)

    keypointsCV1 = []
    keypointsCV2 = []
    for i in range(len(keypoints1)):
        keypointsCV1.append(cv2.KeyPoint(int(keypoints1[i][0]), int(keypoints1[i][1]), 1))
        
    for i in range(len(keypoints2)):
        keypointsCV2.append(cv2.KeyPoint(int(keypoints2[i][0]), int(keypoints2[i][1]), 1))

    image1 = cv2.drawKeypoints(frame1, keypointsCV1, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    image2 = cv2.drawKeypoints(frame2, keypointsCV2, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    axes[0].imshow(image1)
    axes[0].axis('off') 
    axes[1].imshow(image2)
    axes[1].axis('off') 

    plt.show()

def show_homogaphies_given_feat_matches(src, dest, H, video_path, index1, index2):
    """
    Given 2 frames, show the difference between the homography computed by
    our algorithm and the homography computed by opencv.
    
    Parameters
    ----------
    src : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the first image that have a corresponding point in the second image.
    dest : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the second image that have a corresponding point in the first image.
    H : numpy.ndarray
        Array of shape (3, 3) representing the homography matrix.
    video_path : str
        Path to the video file.
    index1 : int
        Index of the first frame.
    index2 : int
        Index of the second frame.
    """
    Hgood, _ = cv2.findHomography(src, dest)

    image = get_single_frame(video_path,index1)
    image2 = get_single_frame(video_path,index2)

    transformed_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    transformed_imageCV = cv2.warpPerspective(image, Hgood, (image.shape[1], image.shape[0]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10)) 
    axes[0, 0].imshow(image) #if amogus = sussy else axes[0, 0].imshow(image) # sus      
    axes[0, 0].axis('off') 
    axes[0, 1].imshow(image2)
    axes[0, 1].axis('off') 
    axes[1, 0].imshow(transformed_image)
    axes[1, 0].axis('off') 
    axes[1, 1].imshow(transformed_imageCV)
    axes[1, 1].axis('off') 
    plt.show()
    
def show_matches(matches1, matches2, match1to2, video_path, frame_index1, frame_index2):
    """ Show the matches between 2 frames using opencv."""
    match1to2_cv = []
    for i in range(len(match1to2)):
        match1to2_cv.append([cv2.DMatch(match1to2[i][1],match1to2[i][0],0)])
    
    kp1 = []
    kp2 = []                
    for i in range(len(matches1)):
        kp1.append(cv2.KeyPoint(int(matches1[i,0]),int(matches1[i,1]),1))
        kp2.append(cv2.KeyPoint(int(matches2[i,0]),int(matches2[i,1]),1))

    frames = [get_single_frame(video_path, frame_index1), get_single_frame(video_path, frame_index2)]
    img = cv2.drawMatchesKnn(frames[0], tuple(kp1), frames[1], tuple(kp2), match1to2_cv, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(img)
    plt.show

def show_homogaphies(H, video_path, index1, index2):
    """
    Given 2 frames, show the difference between the homography computed by
    our algorithm and the homography computed by opencv.
    
    Parameters
    ----------
    src : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the first image that have a corresponding point in the second image.
    dest : numpy.ndarray
        Array of shape (N, 2) where we have the keypoints from the second image that have a corresponding point in the first image.
    H : numpy.ndarray
        Array of shape (3, 3) representing the homography matrix.
    video_path : str
        Path to the video file.
    index1 : int
        Index of the first frame.
    index2 : int
        Index of the second frame.
    """
    

    image = get_single_frame(video_path,index1)
    image2 = get_single_frame(video_path,index2)

    sift = cv2.SIFT_create(2000)
    keypoints1, descriptors1 = sift.detectAndCompute(image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)


    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)


    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    

    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


    transformed_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    transformed_imageCV = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10)) 
    axes[0, 0].imshow(image)
    axes[0, 0].axis('off') 
    axes[0, 1].imshow(image2)
    axes[0, 1].axis('off') 
    axes[1, 0].imshow(transformed_image)
    axes[1, 0].axis('off') 
    axes[1, 1].imshow(transformed_imageCV)
    axes[1, 1].axis('off') 
    plt.show()



def draw_on_map(map_path, homographies):

    # Read the image
    img = mpimg.imread(map_path)
    height, width = img.shape[:2]

    # Camera position (fixed)
    cam_pos = np.array([width/2, height/2, 1])
    x = [width/2]
    y = [height/2]

    for i in range(len(homographies)):
        
        transf = np.identity(3)
        for j in range(i-1,0,-1):
            transf = transf @ homographies[j]
            
        point = transf @ cam_pos                # Coordinates in homogeneous form
        x.append(float(point[0]/point[2]))      # Divide by the last to convert to cartesian
        y.append(float(point[1]/point[2]))      # Divide by the last to convert to cartesian

    # Display the image using imshow
    plt.imshow(img)
    plt.scatter(x, y, color='red', s=1)
    plt.scatter(x[0], y[0], marker="x", color='#2e8bc0', s=30)
    plt.scatter(x[-1], y[-1], marker="x", color='#2e8bc0', s=30)
    plt.text(x[0]+20, y[0]-10, 'Start', fontsize=9, verticalalignment='bottom', horizontalalignment='right', color='white')
    plt.text(x[-1]+15, y[-1]+10, 'End', fontsize=9, verticalalignment='top', horizontalalignment='right', color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return