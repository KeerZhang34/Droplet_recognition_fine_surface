import os.path
import glob
import cv2
import numpy as np
import image_processing_in_batch.separated_processing.a_read_file_batch as a_read_file_batch
import image_processing_in_batch.separated_processing.b_pre_processing as b_pre_processing
import image_processing_in_batch.separated_processing.c_binarizing as c_binarizing
import image_processing_in_batch.separated_processing.d_DE_and_filling as d_DE_and_filling
import image_processing_in_batch.separated_processing.e_labeling_and_plotting as e_labeling_and_plotting



def all_together(file_path='', background_image_path='',binarizing_mono=False, DE=False,
                 clear_pre_saved_imgs=True,
                 ada_win_size=80):


    file_name=file_path.split(sep='\\')[-1]
    folder_path='\\'.join(file_path.split(sep='\\')[:-1])
    background_image_name='62000_8932.45.bmp'

    #folder_path_to_save = 'C:\\Users\\Lenovo\\Desktop\\image_results\\' + file_name[:-4]
    folder_path_to_save=folder_path+'\\'+file_name[:-4]


    if not os.path.isdir(folder_path_to_save):
        os.makedirs(folder_path_to_save)

    if clear_pre_saved_imgs:
        files = glob.glob(folder_path_to_save + '\\*')
        for f in files:
            os.remove(f)






    file_handler=cv2.imread(file_path).astype(np.uint8)

    # ori_contrast: uint8; gray_scale_ori: uint 8
    #ori_contrast = b_pre_processing.enhancing_contrats(file_handler)
    gray_scale_ori = cv2.cvtColor(file_handler, cv2.COLOR_BGR2GRAY)



    # file_handler_2: sharpened image, datatype: uint8
    remove_background=False

    if remove_background:
        background_image = cv2.imread(background_image_path).astype(np.uint8)
        # same datatype for the following two files
        background_contrast = b_pre_processing.enhancing_contrats(background_image)
        gray_scale_background_image = cv2.cvtColor(background_contrast, cv2.COLOR_BGR2GRAY)
        filer_handler_2=b_pre_processing.preprocessing(gray_scale_ori,gray_scale_background_image,
                                                       frame_ID='image',
                                                       folder_path=folder_path_to_save)
    else:
        filer_handler_2 = b_pre_processing.preprocessing(gray_scale_ori,
                                                         gray_scale_background_image=None,
                                                         frame_ID='image',
                                                         background_removing=remove_background)

    # binarized: uint8


    if binarizing_mono:
        _, binarized=c_binarizing.binarizing_mono(frame_ID='image',
                                                  file_handler=filer_handler_2,
                                                  write=True,
                                                  folder_path=folder_path_to_save,
                                                  threshold_method=cv2.THRESH_BINARY+cv2.THRESH_OTSU,
                                                  threshold=0, # referred when method == cv2.THRESH_BINARY
                                                  threshold_diff=-15) # referred when method == cv2.THRESH_BINARY + cv2.THRESH_OTSU
    else:
        binarized=c_binarizing.binarizing_adaptive(ada_win_size,frame_ID='image',
                                                   file_handler=filer_handler_2,
                                                   extracted_threshold_value=-5,
                                                   write=True,
                                                   folder_path=folder_path_to_save,
                                                   addition='_window_size_{}'.format(ada_win_size))



    # dilate and erode. only apply when really needed. for_filling: uint8


    if DE:
        for_filling=d_DE_and_filling.opening(binarized,5,
                                             frame_ID='image',
                                             write=True,
                                             folder_path=folder_path_to_save,
                                             binarizing_mono=binarizing_mono)
    else:
        for_filling=binarized


    # before filling, draw rectangle and ramove corner pixel
    # with_rectangle_corner: uint8
    with_rectangle_corner=d_DE_and_filling.draw_rectangle_and_corner(input_file=for_filling,
                                                                     line_thickness=3,
                                                                     corner_pixel_removed=5)


    # filling and saving file
    # filled: uint8
    filled=d_DE_and_filling.filling(file_handler=with_rectangle_corner,
                                    frame_ID='image',
                                    folder_path=folder_path_to_save,
                                    binarizing_mono=binarizing_mono,
                                    addition='_window_size_{}'.format(ada_win_size))

    # remove added rectangle and corner
    rectangle_corner_inverse=d_DE_and_filling.draw_rectangle_corner_inverse(file_handler=filled,line_thickness=5)

    # labelling and plotting
    droplet_size_value, real_area, coverage=e_labeling_and_plotting.pattern_recognition(
        file_handler=rectangle_corner_inverse,
        frame_ID='image',
        minimum_size=10,
        binarizing_mono=binarizing_mono,
        write_labelling=True,
        folder_path=folder_path_to_save,
        addition='_window_size_{}'.format(ada_win_size))

    e_labeling_and_plotting.droplet_statistic(droplet_size_list=droplet_size_value,
                                              frame_ID='images',
                                              local_time='',
                                              coverage=coverage,
                                              plot_statistic=True,
                                              write_to_xlsx=True,
                                              write_to_csv=False,
                                              max_droplet_size_for_fixed=10000,
                                              folder_path=folder_path_to_save,
                                              binarizing_mono=binarizing_mono,
                                              addition='_window_size_{}'.format(ada_win_size))




if __name__ == '__main__':

    #
    #C:\\Users\\Lenovo\\Downloads\\20000_4790.60.bmp
    all_together(file_path='E:\\image_ER_env_data_raw\\2021_10_26\\images_26_10_2021\\condensation_1252.72+4600_21_10_2021.bmp',
                 binarizing_mono=False,
                 DE=False,
                 clear_pre_saved_imgs=True,
                 ada_win_size=300)




