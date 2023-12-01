
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
import cv2 as cv
import matplotlib.pyplot as plt

from utils import read
from utils.feat_manipulation import feat_matching
from utils.debug_funcs import show_image_and_features, get_single_frame, show_homogaphies, compare_process_video_sift
from utils.homography import homography

debug = True

def check_if_drawmatches_works(img1, img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    return kp1, kp2, good

def main(config_file, feat_file = 'surf_features.mat'):

    # Read the config file
    if config_file is not None:
        config = read.parse_config_file(config_file)
        map_or_all = config['transforms'][0][1]
        mat_path = config['keypoints_out'][0][0]
        out_path = config['transforms_out'][0][0]

    if debug:
        mat_path = os.path.join(os.path.dirname(__file__), 'data', 'surf_features.mat')
        map_or_all = 'map'

    f = loadmat(mat_path)
    feat = f['features']    
    feat=feat.squeeze() # remove the extra dimension
    print(feat.shape)
    print(feat[0].shape)
    print(feat[0][0].shape)
    print(type(feat))
    print(type(feat[0]))
    print(type(feat[0][0].shape))
    print(feat.dtype)
    print(feat[0].dtype)
    print(feat[0][0].dtype)



    transforms_out_all = np.empty((0,11))
    
    if map_or_all == 'map':
        for k in range(1,len(feat)):
            transforms_out = np.array([])

            matches1, matches2, match1to2 = feat_matching(feat[k], feat[k-1])
            
            if debug:
                #compare_process_video_sift("src/data/backcamera_s1.mp4", "src/data/surf_features.mat", k)
                #show_image_and_features("src/data/backcamera_s1.mp4", k, k-1, matches1, matches2) # show the keypoints to make sure they make sense
                
                match1to2_cv = []
                for i in range(len(match1to2)):
                    match1to2_cv.append([cv.DMatch(match1to2[i][1],match1to2[i][0],0)])
                
                kp1 = []
                kp2 = []                
                for i in range(len(matches1)):
                    kp1.append(cv.KeyPoint(int(matches1[i,0]),int(matches1[i,1]),1))
                    kp2.append(cv.KeyPoint(int(matches2[i,0]),int(matches2[i,1]),1))

                print(matches1)


                #kp1_good, kp2_good, matches_good = check_if_drawmatches_works(np.uint8(get_single_frame("src/data/backcamera_s1.mp4", k)), np.uint8(get_single_frame("src/data/backcamera_s1.mp4", k-1)))
 

                frames = [np.uint8(get_single_frame("src/data/backcamera_s1.mp4", k)), np.uint8(get_single_frame("src/data/backcamera_s1.mp4", k-1))]
                img = cv.drawMatchesKnn(frames[0], tuple(kp1), frames[1], tuple(kp2), match1to2_cv, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.imshow(img)
                plt.show()
            
            H = homography(matches1, matches2)
            print(H)
            #show_homogaphies(matches1, matches2, H, "src/data/backcamera_s1.mp4", 0, i)

            transforms_out = np.concatenate((transforms_out, np.asarray([0,i])))
            transforms_out = np.concatenate((transforms_out, H.reshape(9)))

            transforms_out_all = np.vstack((transforms_out_all,transforms_out))
    transforms_out_all = transforms_out_all.T
    

    out_mat_format = {"matrix": transforms_out_all, "label": "transforms out"}
    savemat(out_path, out_mat_format)



if __name__ == '__main__':
    #main(sys.argv[1])
    main(None)