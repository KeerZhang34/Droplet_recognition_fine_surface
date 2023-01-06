import os
import numpy as np
import cv2
from image_processing_in_batch.separated_processing.c_binarizing import binarizing_adaptive, binarizing_mono
from image_processing_in_batch.separated_processing.a_read_single_file import read_single_file

#binarized=binarizing_adaptive(12)
#binarized=binarizing_mono(0)
#file_name=read_single_file(0)[-1]



def draw_rectangle_and_corner(input_file,line_thickness,corner_pixel_removed):

    #input file: uint8
    h, w = input_file.shape[:2]
    color = 255 # white

    # draw a white rectangle at very outside

    upper_left = (0, 0)
    bottom_right = (w, h)
    file_handler_rect = cv2.rectangle(input_file, upper_left, bottom_right, color, line_thickness)
    h1, w1 = file_handler_rect.shape[:2]
    # print(h1, w1)

    # remove edge pixels to ensure biggest rectangle not filled

    for i in range(0, corner_pixel_removed):
        for j in range(0, corner_pixel_removed):
            # file_handler_rect[x,y]: x: row; y:column
            file_handler_rect[i, j] = 0
    for i in range(h1 - corner_pixel_removed, h1):
        for j in range(0, corner_pixel_removed):
            file_handler_rect[i, j] = 0
    for i in range(0, corner_pixel_removed):
        for j in range(w1 - corner_pixel_removed, w1):
            file_handler_rect[i, j] = 0
    for i in range(h1 - corner_pixel_removed, h1):
        for j in range(w1 - corner_pixel_removed, w1):
            file_handler_rect[i, j] = 0

    # remove another 16 random spots to ensure the biggest blank will not be close
    #h_ran: upper left pixel
    h_ran=np.random.randint(low=corner_pixel_removed,high=h1-2*corner_pixel_removed,size=8)

    w_ran=np.random.randint(low=corner_pixel_removed,high=w1-2*corner_pixel_removed,size=8)

    # h_ran/2 spots on left edge
    for index in range(int(len(h_ran)/2)):
        for i in range(h_ran[index], h_ran[index]+corner_pixel_removed):
            for j in range(0, corner_pixel_removed):
            # file_handler_rect[x,y]: x: row; y:column
                file_handler_rect[i, j] = 0
    # h_ran/2 spots on right edge
    for index in range(int(len(h_ran)/2)):
        for i in range(h_ran[-index-1], h_ran[-index-1]+corner_pixel_removed):
            for j in range(w1 - corner_pixel_removed, w1):
                file_handler_rect[i, j] = 0
    # w_ran/2 spots on upper edge
    for index in range(int(len(w_ran)/2)):
        for i in range(0, corner_pixel_removed):
            for j in range(w_ran[index],w_ran[index]+corner_pixel_removed):
                file_handler_rect[i, j] = 0
    # w_ran/2 spots on bottom edge
    for index in range(int(len(w_ran)/2)):
        for i in range(h1 - corner_pixel_removed, h1):
            for j in range(w_ran[-index-1],w_ran[-index-1]+corner_pixel_removed):
                file_handler_rect[i, j] = 0



    return file_handler_rect


def kernal(n):
    kernal_processing=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*n+1,2*n+1))
    return kernal_processing

def opening(file,n,frame_ID,write=True, folder_path='', binarizing_mono=True):
    opening_processing=cv2.morphologyEx(file,cv2.MORPH_OPEN,kernal(n))
    if write:
        os.chdir(folder_path)
        cv2.imwrite('{}_opening_mono.bmp'.format(frame_ID),opening_processing) if binarizing_mono \
            else cv2.imwrite('{}_opening_binarized.bmp'.format(frame_ID),opening_processing)

    return opening_processing

def closing(file,n,frame_ID,write=True, folder_path='', binarizing_mono=True):
    closing_processing=cv2.morphologyEx(file,cv2.MORPH_CLOSE,kernal(n))
    if write:
        os.chdir(folder_path)
        cv2.imwrite('{}_closing_mono.bmp'.format(frame_ID), closing_processing) if binarizing_mono \
            else cv2.imwrite('{}_closing_adaptive.bmp'.format(frame_ID), closing_processing)
    return closing_processing

def dilate(file,n,iterations):
    dilating_processing=cv2.dilate(file,kernal(n),iterations)
    return dilating_processing

def erode(file,n,iterations):
    eroding_processing=cv2.erode(file,kernal(n),iterations)
    return eroding_processing

def DE(file_handler):
    hole_removed=opening(file_handler,1)
    #post_DE=dilate(erode(hole_removed,1,3),2,3)
    #return hole_removed
    #cv2.imshow('',post_DE)
    #cv2.waitKey(0)
    return hole_removed


def filling(file_handler,frame_ID, folder_path='', binarizing_mono=True, addition='', write_filled=True):
    contours, hierarchy = cv2.findContours(file_handler, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    binarized_copy=file_handler.copy() #perform filling action on binarized_copy
    cv2.drawContours(binarized_copy,contours,-1,255,-1)

    os.chdir(folder_path)
    if binarizing_mono:
        if write_filled:
            cv2.imwrite('{}_filled_mono.bmp'.format(frame_ID),binarized_copy)
    else:
        if write_filled:
            cv2.imwrite('{}_filled_adaptive{}.bmp'.format(frame_ID,addition), binarized_copy)


    return binarized_copy



def draw_rectangle_corner_inverse(file_handler,line_thickness):

    #draw the outer rectangle
    h2, w2 = file_handler.shape[:2]

    color =0
    upper_left = (0, 0)
    bottom_right = (w2, h2)

    file_handler_rect=cv2.rectangle(file_handler,upper_left,bottom_right,color,line_thickness)

    return file_handler_rect





if __name__ == '__main__':
    filling(0)
    #cv2.imshow('',filling(0))
    #cv2.waitKey(0)

