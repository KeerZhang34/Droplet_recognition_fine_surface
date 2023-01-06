import cv2
import os
from image_processing_in_batch.separated_processing.b_pre_processing import preprocessing
from image_processing_in_batch.separated_processing.a_read_single_file import read_single_file


def binarizing_adaptive(n,frame_ID,file_handler,extracted_threshold_value,write=True, folder_path='', addition=''):

    blur = cv2.GaussianBlur(file_handler, (5, 5), 0)
    binarized=cv2.adaptiveThreshold(file_handler,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,2*n+1,extracted_threshold_value)
    #img=cv2.bitwise_not(binarized)

    if write:

        os.chdir(folder_path)
        cv2.imwrite('{}_binarized_adaptive{}.bmp'.format(frame_ID, addition), binarized)



    return binarized


def binarizing_mono(frame_ID,file_handler,threshold_method=cv2.THRESH_BINARY + cv2.THRESH_OTSU,threshold=0,
                    threshold_diff=0,write=True, folder_path=''):

    blur = cv2.GaussianBlur(file_handler,(5,5),0)

    if threshold_method==cv2.THRESH_BINARY:
        thresh, binarized = cv2.threshold(blur, threshold, 255, threshold_method)
    else: # threshold_method==cv2.THRESH_BINARY + cv2.THRESH_OTSU
        thresh_otsu, _ = cv2.threshold(blur, 0, 255, threshold_method)
        thresh, binarized= cv2.threshold(blur, thresh_otsu+threshold_diff, 255, cv2.THRESH_BINARY)

    if write:
        os.chdir(folder_path)
        cv2.imwrite('{}_binarized_mono.bmp'.format(frame_ID), binarized)

    return thresh, binarized
