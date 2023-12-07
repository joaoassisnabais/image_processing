
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat

from utils.read import parse_config_file
from utils.debug_funcs import *
from utils.feat_manipulation import feat_matching, RANSAC
from utils.homography import homography

debug = True

def main(config_file):

    # Read the config file
    if config_file is not None:
        config = parse_config_file(config_file)
        video_path = config['videos'][0][0]
        map_or_all = config['transforms'][0][1]
        mat_path = config['keypoints_out'][0][0]
        out_path = config['transforms_out'][0][0]
    
    #while we have no config file, use the default values
    #mat_path = os.path.join(os.path.dirname(__file__), 'out', 'features.mat')
    #map_or_all = 'map'

    f = loadmat(mat_path)
    feat = f['features']    
    feat=feat.squeeze() # remove the extra dimension

    transforms_out_all = np.empty((0,11))
    store_H = [np.identity(3)]  # index k -> homography between frame k and previus frame
    
    for k in range(1, len(feat)):
        transforms_out = np.array([])

        matches1, matches2, matches1to2 = feat_matching(feat[k], feat[k-1])
        
        if debug:
            show_image_and_keypoints(video_path, k, k-1, matches1, matches2)
            show_matches(matches1, matches2, matches1to2, video_path, 0, k)
        
        matches1, matches2, _ = RANSAC(matches1, matches2)
        
        H = homography(matches1, matches2)
        
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
            show_homogaphies(matches1, matches2, H, video_path, 0, k)

            transforms_out = np.concatenate((transforms_out, np.asarray([0,k])))
            transforms_out = np.concatenate((transforms_out, H.reshape(9)))

            transforms_out_all = np.vstack((transforms_out_all,transforms_out))

    if map_or_all == 'all':
        for i in range(1,len(store_H)):
            
            for j in range(i+1, len(store_H)):
                H = np.matmul(H, np.linalg.inv(store_H[k-1]))

                show_homogaphies(matches1, matches2, H, video_path, 0, k)

                transforms_out = np.array([])
                transforms_out = np.concatenate((transforms_out, np.asarray([i,j])))
                transforms_out = np.concatenate((transforms_out, H.reshape(9)))

                transforms_out_all = np.vstack((transforms_out_all,transforms_out))


    transforms_out_all = transforms_out_all.T
    

    out_mat_format = {"matrix": transforms_out_all, "label": "transforms out"}
    savemat(out_path, out_mat_format)


if __name__ == '__main__':
    main(sys.argv[1])
    