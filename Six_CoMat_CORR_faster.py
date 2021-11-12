"KK"
import numpy as np
import cv2
from glob import glob



#much faster implementation of the co-occurrence matrix computation
def cooccurence_fast(image1, image2, distance_x, distance_y, levels):

    fast_cooccurence_matrix_D = np.zeros((levels,levels), dtype='int64') # Preallocate for Diognal case

    #prepare images for matricial operations
    sx = np.size(image1, 0)
    sy = np.size(image1, 1)
    image1_ready = image1[0:sx-distance_x,0:sy-distance_y]
    image2_ready = image2[distance_x:, distance_y:]


    for i in range(levels):
        image2_ready_temp= image2_ready[image1_ready == i]
        for j in range(levels):

                fast_cooccurence_matrix_D[i,j] = np.sum(image2_ready_temp == j)

    return fast_cooccurence_matrix_D


j = 0
for i in range(0, 1):

    print('-------------------------------------------------------------------- ')
    print('Read the images from folder ({}) for computing six Co-matrices .... '.format(i))
    print('-------------------------------------------------------------------- ')
    
    image_glob = sorted(glob(r'/Volumes/My1TBDrive/FF++/extracted_frames/manipulated/*'))
    for idx, im_file in enumerate(image_glob):

        image = cv2.imread(im_file, cv2.COLOR_BGR2RGB)
        im_file = im_file.split("/")
        file_name = im_file[-1]
        file_name = file_name.replace(".jpg", "")

        first_digits = int(file_name[0:3])
        
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]

        levels = 256

        ###############################VIPP and USB##############################################
        # Horizontal
        distance_x = 1
        distance_y = 1



        RR_D = cooccurence_fast(R, R, distance_x, distance_y, levels)
        GG_D = cooccurence_fast(G, G, distance_x, distance_y, levels)
        BB_D = cooccurence_fast(B, B, distance_x, distance_y, levels)


        distance_xx = 0
        distance_yy = 0



        RG_D = cooccurence_fast(R, G, distance_xx, distance_yy, levels)
        RB_D = cooccurence_fast(R, B, distance_xx, distance_yy, levels)
        GB_D = cooccurence_fast(G, B, distance_xx, distance_yy, levels)




        # VIPP
        tensor1 = np.stack((RR_D, GG_D, BB_D, RG_D, RB_D, GB_D))

        tensorVIPP = np.swapaxes(tensor1, 0, 2)

        np.save('/Volumes/My1TBDrive/FF++/dataset_co/manipulated/%s.npy' % file_name, tensorVIPP)

        print('RAW_VIPP: Image number ({}) from folder ({})... '.format(j, i))

        j = j + 1
