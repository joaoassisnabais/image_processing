import sys
import os

import scipy.io as sio

mat_path = os.path.join(os.path.dirname(__file__), 'data', 'back_camera_pinhole.mat')

mat = sio.loadmat(mat_path)

print(mat)