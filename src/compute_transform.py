
import os
import sys
import numpy as np
from scipy.io import loadmat
from utils import read

from utils.feat_manipulation import match_feats


def main(feat_file = 'surf_features.mat'):


    mat_path = os.path.join(os.path.dirname(__file__), 'data', 'surf_features.mat')

    f = loadmat(mat_path)
    feat = f['features']    
    feat=feat.squeeze() # remove the extra dimension

    match_feats(feat[0], feat[1])
        
        
    

if __name__ == '__main__':
    main()