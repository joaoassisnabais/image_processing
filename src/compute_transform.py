
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
from utils import read

from utils.feat_manipulation import feat_matching
from utils.debug_funcs import show_image_and_features, get_single_frame, show_homogaphies
from utils.homography import homography



def main(config_file, feat_file = 'surf_features.mat'):

    # Read the config file
    config = read.parse_config_file(config_file)

    map_or_all = config['transforms'][0][1]
    mat_path = config['keypoints_out'][0][0]
    out_path = config['transforms_out'][0][0]
    #mat_path = os.path.join(os.path.dirname(__file__), 'data', 'surf_features.mat')

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
        for i in range(1,len(feat)):
            transforms_out = np.array([])

            matches1, matches2, match1to2 = feat_matching(feat[i], feat[0])
            show_image_and_features("src/data/backcamera_s1.mp4", 1, 2, matches1, matches2) # show the keypoints to make sure they make sense
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
    main(sys.argv[1])