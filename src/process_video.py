import os
import sys
from scipy.io import loadmat
from utils import read

def main(config_file):

    config_path = os.path.join(os.path.dirname(__file__), 'data', config_file)

    # Read the config file
    config = read.parse_config_file(config_path)



if __name__ == '__main__':
    main(sys.argv[1])
