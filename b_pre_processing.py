import os

import cv2
import numpy as np
from image_processing_in_batch.separated_processing.a_read_single_file import read_single_file
import image_processing_in_batch.separated_processing.a_read_file_batch as a_read_file_batch

#ori_file=read_single_file(0)[0]
#background_image=cv2.imread('C:\\Users\\kzhang9\\Desktop\\15_02_2022_3\\44000+93441.81.bmp')


# enhancing contrast
def enhancing_contrats(file, cliplimit=3.0, tileGridSize=(0,0)):
    lab = cv2.cvtColor(file, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tileGridSize)
    enhanced_l = clahe.apply(l)
    lab_enhanced = cv2.merge((enhanced_l, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


#ori_contrast=enhancing_contrats(ori_file)
#gray_scale_ori = cv2.cvtColor(ori_contrast, cv2.COLOR_BGR2GRAY)

#background_contrast=enhancing_contrats(background_image)
#gray_scale_background_image=cv2.cvtColor(background_contrast,cv2.COLOR_BGR2GRAY)








def background_removal(gray_scale_ori,gray_scale_background_image,frame_ID, write, folder_path):
    #gray_scale_ori: unit8
    #gray_scale_background_image: uint8

    h,w=gray_scale_ori.shape[:2]


    background_removed = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if gray_scale_ori[i,j] >= gray_scale_background_image[i,j]:
                background_removed[i,j] = gray_scale_ori[i,j] - gray_scale_background_image[i,j]
            else:
                continue

    if write:
        os.chdir(folder_path)
        cv2.imwrite('{}_background_removed.bmp'.format(frame_ID),background_removed)



    return background_removed

def preprocessing(gray_scale_ori,gray_scale_background_image,frame_ID,background_removing=True, folder_path=''):


    #input file: sfter enhancing contrast

    if background_removing:
        input_file=background_removal(gray_scale_ori,gray_scale_background_image,frame_ID, write=True, folder_path=folder_path)
    else:
        input_file = gray_scale_ori




    # denoising --> sharpening --> binarizing

    denoised = cv2.fastNlMeansDenoising(input_file, None, 10, 7, 21)
    kernal = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernal)


    return sharpened

if __name__ == '__main__':
    cv2.imshow('',preprocessing(10,12))
    cv2.waitKey(0)