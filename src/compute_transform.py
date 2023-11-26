
import os
import sys
import numpy as np
from scipy.io import loadmat
from utils import read

from utils.feat_manipulation import feat_matching
from utils.debug_funcs import show_image_and_features, get_single_frame, show_homogaphies
from utils.homography import homography



def main(config_file, feat_file = 'surf_features.mat'):

    # Read the config file
    config = read.parse_config_file(config_file)

    mat_path = config['keypoints_out'][0][0]
    #mat_path = os.path.join(os.path.dirname(__file__), 'data', 'surf_features.mat')

    f = loadmat(mat_path)
    feat = f['features']    
    feat=feat.squeeze() # remove the extra dimension

    matches1, matches2, match1to2 = feat_matching(feat[1], feat[2])

    #show_image_and_features("src/data/backcamera_s1.mp4", 1, 2, matches1, matches2) # show the keypoints to make sure they make sense

    H = homography(matches1, matches2)

    show_homogaphies(matches1, matches2, H, "src/data/backcamera_s1.mp4")


if __name__ == '__main__':
    main(sys.argv[1])