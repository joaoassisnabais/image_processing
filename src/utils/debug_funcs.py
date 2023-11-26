import cv2
from matplotlib import pyplot as plt


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


def show_image_and_features(video_path, index1, index2, keypoints1, keypoints2):

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


def show_homogaphies(src, dest, H, video_path, index1, index2):
    Hgood, _ = cv2.findHomography(src, dest)

    image = get_single_frame(video_path,index1)
    image2 = get_single_frame(video_path,index2)


    transformed_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    transformed_imageCV = cv2.warpPerspective(image, Hgood, (image.shape[1], image.shape[0]))

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