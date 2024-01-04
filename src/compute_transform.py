
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat

from utils.read import parse_config_file, convert_file_path
from utils.debug_funcs import *
from utils.feat_manipulation import feat_matching, RANSAC
from utils.homography import homography

debug = True

def main(config_file):

    # Read the config file
    if config_file is not None:
        config = parse_config_file(config_file)
        video_path = convert_file_path(config['videos'][0][0])
        map_or_all = config['transforms'][0][1]
        mat_path = convert_file_path(config['keypoints_out'][0][0])
        out_path = convert_file_path(config['transforms_out'][0][0])

        given_matches_map = config['pts_in_map']
        given_matches_video = config['pts_in_frame']


    #while we have no config file, use the default values
    mat_path = os.path.join(os.path.dirname(__file__), 'out', 'features.mat')
    #map_or_all = 'map'

    f = loadmat(mat_path)
    feat = f['features']    
    feat=feat.squeeze() # remove the extra dimension

    transforms_out_all = np.empty((0,11))
    store_H = [np.identity(3)]  # index k -> homography between frame k and previus frame
    store_k_to_map = [] # index k -> homography between frame k and the map

    given_k_to_map_homographies = [] # homographies between given frames and the map
    given_k_to_map_frames = [] # index k -> given_k_to_map_homographies[k] is the homography between frame k and the map

    store_matches = [[0,0],]  # used for debug. index k -> feature matches between frame k and the previus frame
    
    
    for k in range(1, len(feat)):
        transforms_out = np.array([])

        matches1, matches2, matches1to2 = feat_matching(feat[k], feat[k-1])
        
        if debug:
            show_image_and_keypoints(video_path, k, k-1, matches1, matches2)
            #show_matches(matches1, matches2, matches1to2, video_path, 0, k)
        
        matches1, matches2, _ = RANSAC(matches1, matches2)

        H = homography(matches1, matches2)

        #show_homogaphies_given_feat_matches(matches1, matches2, H, video_path, k, k-1)   
     

        #obtain the homography from current frame to the previus frame
        store_H += [np.copy(H)]



        
    if map_or_all == 'map':
        #calculate the homographies using the given points
        for i in range(len(given_matches_map)):
            points_map = []
            points_frames = []
            for j in range(1,len(given_matches_map[i]),2):
                points_map += [[int(given_matches_map[i][j]),int(given_matches_map[i][j+1])]]
                points_frames += [[int(given_matches_video[i][j]),int(given_matches_video[i][j+1])]]

            given_k_to_map_homographies += [homography(np.asarray(points_frames), np.asarray(points_map))]
            given_k_to_map_frames += [int(given_matches_video[i][0]) - 1]

        print(given_k_to_map_homographies)
        print(given_k_to_map_frames)



        # initialize store_k_to_map
        for i in range(len(store_H)):
            store_k_to_map += [np.identity(3)]
        
        # fill in store_k_map until the first given frame
        first_frame = given_k_to_map_frames[0]
        first_H = given_k_to_map_homographies[0]
        store_k_to_map[first_frame] = np.copy(first_H)

        for i in range(first_frame-1, -1, -1):
            H = np.matmul(store_k_to_map[i+1], np.linalg.inv(store_H[i+1])) # H_k+1_to_map * H_k_to_k+1 
            store_k_to_map[i] = np.copy(H)
   

            
        

        # calculate the homography from each frame to the map, using the given frames
        for i in range(first_frame+1, len(store_H)):
            if(i in given_k_to_map_frames):
                H = given_k_to_map_homographies[given_k_to_map_frames.index(i)]
            else:
                H = np.matmul(store_k_to_map[i-1], store_H[i]) # H_k-1_to_map * H_k_to_k-1 #np.matmul(H, np.linalg.inv(store_H[k-1]))
                        
            store_k_to_map[i] = np.copy(H)
     
                    
        
        for i in range(len(store_k_to_map)):   
            #print(store_k_to_map[i])

            #save homographies in the output format 
            transforms_out = np.array([])
            transforms_out = np.concatenate((transforms_out, np.asarray([0,i])))
            transforms_out = np.concatenate((transforms_out, store_k_to_map[i].reshape(9)))

            transforms_out_all = np.vstack((transforms_out_all,transforms_out))

    elif map_or_all == 'all':
        
        #calculate the homography from current frame to all next frames
        for i in range(1,len(store_H)-1):
            H_i_to_previous_j = np.identity(3)
            for j in range(i+1, len(store_H)):
                if(j == i+1):
                    H = np.linalg.inv(store_H[j])
                    H_i_to_previous_j = np.copy(H)
                else:
                    H = np.matmul(np.linalg.inv(H_i_to_previous_j), store_H[j]) # H_j-1_to_i * H_j_to_j-1
                    H_i_to_previous_j = np.copy(H)

                 #H = np.matmul(H, np.linalg.inv(store_H[k-1]))

                #show_homogaphies(H, video_path, i, j)

                #save homographies in the output format 
                transforms_out = np.array([])
                transforms_out = np.concatenate((transforms_out, np.asarray([i,j])))
                transforms_out = np.concatenate((transforms_out, H.reshape(9)))

                transforms_out_all = np.vstack((transforms_out_all,transforms_out))



    transforms_out_all = transforms_out_all.T
    print(transforms_out_all)

    out_mat_format = {"matrix": transforms_out_all, "label": "transforms out"}
    savemat(out_path, out_mat_format)


if __name__ == '__main__':
    main(sys.argv[1])
    