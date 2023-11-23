
import os
import sys
from scipy.io import loadmat
from utils import read

only_transform = True

def main(config_file):

    if only_transform:
        mat_path = os.path.join(os.path.dirname(__file__), 'data', 'surf_features.mat')

        f = loadmat(mat_path)
        feat = f['features']

    # Read the config file
    config = read.parse_config_file(config_file)



if __name__ == '__main__':
    main(sys.argv[1])
