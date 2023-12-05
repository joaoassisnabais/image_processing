
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
import cv2 as cv
import matplotlib.pyplot as plt

from utils import read
from utils.feat_manipulation import feat_matching, RANSAC
from utils.debug_funcs import show_image_and_features, get_single_frame, show_homogaphies, compare_process_video_sift, check_if_drawmatches_works
from utils.homography import homography

debug = False

def main(config_file):

    # Read the config file
    if config_file is not None:
        config = read.parse_config_file(config_file)
        map_or_all = config['transforms'][0][1]
        mat_path = config['keypoints_out'][0][0]
        out_path = config['transforms_out'][0][0]
    
    mat_path = os.path.join(os.path.dirname(__file__), 'out', 'features.mat')
    map_or_all = 'map'

    f = loadmat(mat_path)
    feat = f['features']    
    feat=feat.squeeze() # remove the extra dimension

    transforms_out_all = np.empty((0,11))
    store_H = [np.identity(3)]  # index k -> homography between frame k and previus frame
    
    for k in range(1, len(feat)):
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

            #kp1_good, kp2_good, matches_good = check_if_drawmatches_works(np.uint8(get_single_frame("src/data/backcamera_s1.mp4", k)), np.uint8(get_single_frame("src/data/backcamera_s1.mp4", k-1)))

            frames = [get_single_frame("src/data/backcamera_s1.mp4", k), get_single_frame("src/data/backcamera_s1.mp4", k-1)]
            img = cv.drawMatchesKnn(frames[0], tuple(kp1), frames[1], tuple(kp2), match1to2_cv, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img)
            plt.show
        
        matches1, matches2, mask = RANSAC(matches1, matches2)
        
        show_image_and_features("src/data/backcamera_s1.mp4", k, k-1, matches1, matches2) # show the keypoints to make sure they make sense
        
        H = homography(matches1, matches2)
        print(H)
        
        if map_or_all == 'map':
            #obtain the homography from current frame to map from the homography from current frame to previous frame
            if(k==1):
                store_H += [np.copy(H)]
            else:
                store_H += [np.copy(H)]
                H = np.matmul(H, np.linalg.inv(store_H[k-1]))
                print(len(store_H),k-1)
        elif map_or_all == 'all':

            store_H += [np.copy(H)]
        else:
            print("bad config. file: transforms must be map or all")

        if map_or_all == 'map':
            show_homogaphies(matches1, matches2, H, "src/data/backcamera_s1.mp4", 0, k)

            transforms_out = np.concatenate((transforms_out, np.asarray([0,k])))
            transforms_out = np.concatenate((transforms_out, H.reshape(9)))

            transforms_out_all = np.vstack((transforms_out_all,transforms_out))

    if map_or_all == 'all':
        for i in range(1,len(store_H)):
            
            for j in range(i+1, len(store_H)):
                H = np.matmul(H, np.linalg.inv(store_H[k-1]))

                show_homogaphies(matches1, matches2, H, "src/data/backcamera_s1.mp4", 0, k)

                transforms_out = np.array([])
                transforms_out = np.concatenate((transforms_out, np.asarray([i,j])))
                transforms_out = np.concatenate((transforms_out, H.reshape(9)))

                transforms_out_all = np.vstack((transforms_out_all,transforms_out))


    transforms_out_all = transforms_out_all.T
    

    out_mat_format = {"matrix": transforms_out_all, "label": "transforms out"}
    savemat(out_path, out_mat_format)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
    