import os.path
import glob
import matplotlib.pyplot as plt
import pandas as pd

import cv2
import numpy as np
import image_processing_in_batch.separated_processing.a_read_file_batch as a_read_file_batch
import image_processing_in_batch.separated_processing.b_pre_processing as b_pre_processing
import image_processing_in_batch.separated_processing.c_binarizing as c_binarizing
import image_processing_in_batch.separated_processing.d_DE_and_filling as d_DE_and_filling
import image_processing_in_batch.separated_processing.e_labeling_and_plotting as e_labeling_and_plotting
from datetime import datetime




class Time_reading:

    def __init__(self, file_date: str, batch='', addition=''):
        file_date_1 = file_date.split('_')
        file_date_1 = file_date_1[2] + '_' + file_date_1[1] + '_' + file_date_1[0]

        folder_path = 'E:\\image_ER_env_data_raw\\' + file_date_1 + addition+ '\\images_' + file_date+batch
        file_path=folder_path+'\\time_information_' + file_date+batch
        file = pd.read_csv(file_path, sep=',', header=None)

        folder_path_to_save = 'E:\\images_processed\\' + file_date_1 + addition + '\\images_' + file_date + batch + '\\multiple_images'

        self.file=file
        self.folder_path_to_save=folder_path_to_save
        self.file_date=file_date # dd_mm_yyyy
        self.file_date_1=file_date_1 # yyyy_mm_dd
        self.batch=batch
        self.dataFrame=None


    def time_procesing(self, file_date_all, date_string, year=' 2022', env_start='', ER_start='', plot_TimePoint=False):
        x = self.file[4]
        fmt = '%d-%m-%Y %H:%M:%S'


        local_time = []
        for i in range(len(file_date_all)):
            local_time = x.str.replace(' local time=' + date_string[i], file_date_all[i].replace('_', '-')).str.replace(
                year, '')
            x = local_time

        start = len('Camera(id=DEV_000F314EE51F) acquired Frame(id=')
        frameID = self.file[0].str.slice(start=start)

        # ER_local_file: local_time, relative_time, R, R_ref
        # env_formetted_file: ,local time, ambient temperature, relative humidity, dew point

        relative_time_env=[]
        start_timestamp_env = datetime.timestamp(datetime.strptime(env_start, fmt))
        relative_time_ER=[]
        start_timestamp_ER= datetime.timestamp(datetime.strptime(ER_start, fmt))

        for local_time_point in local_time:
            local_time_point_stamp= datetime.timestamp(datetime.strptime(local_time_point, fmt))
            relative_time_env.append(round((local_time_point_stamp-start_timestamp_env)/3600.0, 3)) # timestamp in h
            relative_time_ER.append(round((local_time_point_stamp-start_timestamp_ER)/3600.0, 3)) # timestamp in h


        df = {'local time': local_time,
              'Frame ID': frameID,
              'relative_env/ h': relative_time_env,
              'relative_ER/ h': relative_time_ER
              }

        self.dataFrame = pd.DataFrame(data=df)

        new_file_path = self.folder_path_to_save+'\\t_extracted_' + self.file_date + self.batch+ '.csv'


        self.dataFrame.to_csv(new_file_path, sep=',')

        if plot_TimePoint:
            fig, ax=plt.subplots()
            #y=np.arange(len(relative_time_ER))
            y=np.ones(len(relative_time_ER))
            if relative_time_ER[0]<relative_time_env[0]:
                axis=ax.scatter(relative_time_ER,y,color='green', label='rela_ER', s=1)
            else:
                axis=ax.scatter(relative_time_env,y, color='orange', label='rela_env', s=1)

            plt.tight_layout()
            plt.show()

        return self.dataFrame

class Other_preparations:
    def __init__(self, file_date='', batch='', addition=''):

        file_date_1 = file_date.split('_')
        file_date_1 = file_date_1[2] + '_' + file_date_1[1] + '_' + file_date_1[0]
        folder_path_to_save = 'E:\\images_processed\\' + file_date_1 + addition + '\\images_' + file_date + batch + '\\multiple_images'

        self.folder_path_to_save=folder_path_to_save

    def background_image_reading(self, background_image_name = ''):

        # if this action should be performed, background image should be stored in multiple_images folder
        background_image_path = self.folder_path_to_save + '\\' + background_image_name
        background_image = cv2.imread(background_image_path).astype(np.uint8)
        # same datatype for the following two files

        background_contrast = b_pre_processing.enhancing_contrats(background_image)

        gray_scale_background_image = cv2.cvtColor(background_contrast, cv2.COLOR_BGR2GRAY)

        return gray_scale_background_image

    def folders_creating(self):

        if not os.path.isdir(self.folder_path_to_save+'\\background_removed'):
            os.makedirs(self.folder_path_to_save+'\\background_removed')
        if not os.path.isdir(self.folder_path_to_save+'\\binarization'):
            os.makedirs(self.folder_path_to_save+'\\binarization')
        if not os.path.isdir(self.folder_path_to_save+'\\filled'):
            os.makedirs(self.folder_path_to_save+'\\filled')
        if not os.path.isdir(self.folder_path_to_save+'\\labelled'):
            os.makedirs(self.folder_path_to_save+'\\labelled')
        if not os.path.isdir(self.folder_path_to_save+'\\DE'):
            os.makedirs(self.folder_path_to_save+'\\DE')
        if not os.path.isdir(self.folder_path_to_save + '\\statistic'):
            os.makedirs(self.folder_path_to_save + '\\statistic')

    def folders_cleaning(self):
        sub_folder_names=['background_removed', 'binarization', 'filled', 'labelled', 'DE', 'statistic']

        for name in sub_folder_names:
            files = glob.glob(self.folder_path_to_save + '\\{}\\*'.format(name.replace("'","")))
            for f in files:
                os.remove(f)


class Single_image_all_together: # each image as an object

    # file date in the format of dd_mm_yyyy
    def __init__(self, t_ID_info, i, file_date: str, batch='', addition=''):

        try:
            local_times = t_ID_info['local time']
            frame_IDs = t_ID_info['Frame ID']
            frame_ID=frame_IDs[i]
            local_time=local_times[i]

            self.frame_ID=frame_ID
            self.local_time=local_time

            file_date_1 = file_date.split('_')
            file_date_1 = file_date_1[2] + '_' + file_date_1[1] + '_' + file_date_1[0]

            # folder_path = 'E:\\image_ER_env_data_raw\\' + file_date_1 + addition+'\\images_' + file_date + batch
            folder_path='E:\\images_processed\\' + file_date_1 + addition+'\\images_' + file_date + batch+'\\multiple_images\\fixed_position_cropped'
            folder_path_to_save='E:\\images_processed\\' + file_date_1 + addition+'\\images_' + file_date + batch+'\\multiple_images'

            self.folder_path = folder_path_to_save
            self.file_date = file_date  # dd_mm_yyyy
            self.file_date_1 = file_date_1  # yyyy_mm_dd
            self.batch = batch
            file_path_handler, file_name = a_read_file_batch.read_figure(folder_path, frame_ID)
            file_handler = cv2.imread(file_path_handler).astype(np.uint8)

            self.ori_file=file_handler

        except IndexError:
            pass
        except FileNotFoundError:
            raise FileNotFoundError




    def single_image_processing(self, remove_background=False, gray_scale_background_image=None,
                                binarizing_mono=True, adaptive_window_size=80, DE_method='', DE_size=5,
                                minimum_area_for_recognition=10, max_droplet_size_for_fixed=2000, if_Otsu_when_mono=False,
                                threshold_when_not_Otsu_but_mono=0,thresh_diff_for_Otsu=0, ada_info=''):

        # ori_contrast: uint8; gray_scale_ori: uint 8
        #ori_contrast = b_pre_processing.enhancing_contrats(self.ori_file)
        gray_scale_ori = cv2.cvtColor(self.ori_file, cv2.COLOR_BGR2GRAY)



        if remove_background:

            filer_handler_2 = b_pre_processing.preprocessing(gray_scale_ori, gray_scale_background_image,
                                                         self.frame_ID, background_removing=remove_background, folder_path=self.folder_path+'\\background_removed')
        else:
            filer_handler_2 = b_pre_processing.preprocessing(gray_scale_ori, gray_scale_background_image=None,
                                                        frame_ID=self.frame_ID,
                                                         background_removing=remove_background, folder_path='')

        # binarized: uint8

        if binarizing_mono:
            threshold_method=cv2.THRESH_BINARY+ cv2.THRESH_OTSU if if_Otsu_when_mono else cv2.THRESH_BINARY

            thresh, binarized = c_binarizing.binarizing_mono(self.frame_ID,
                                                             filer_handler_2,
                                                             threshold_method=threshold_method,
                                                             threshold=threshold_when_not_Otsu_but_mono,
                                                             threshold_diff=thresh_diff_for_Otsu,
                                                             write=True,
                                                             folder_path=self.folder_path+'\\binarization')
        else:
            thresh, binarized = c_binarizing.binarizing_adaptive(adaptive_window_size, frame_ID=self.frame_ID, file_handler=filer_handler_2,
                                                        extracted_threshold_value=0,
                                                         write=True,
                                                         folder_path=self.folder_path+'\\binarization',
                                                         addition='_window_size_{}'.format(adaptive_window_size))


        # dilate and erode. only apply when really needed. for_filling: uint8

        if DE_method=='OP':

            for_filling = d_DE_and_filling.opening(binarized, DE_size,frame_ID=self.frame_ID, write=True,
                                                   folder_path=self.folder_path+'\\DE', binarizing_mono=binarizing_mono)
        elif DE_method=='CL':
            for_filling = d_DE_and_filling.closing(binarized, DE_size,frame_ID=self.frame_ID, write=True,
                                                   folder_path=self.folder_path+'\\DE', binarizing_mono=binarizing_mono)
        else:
            for_filling = binarized

        # before filling, draw rectangle and ramove corner pixel
        # with_rectangle_corner: uint8
        with_rectangle_corner = d_DE_and_filling.draw_rectangle_and_corner(input_file=for_filling,
                                                                           line_thickness=10,
                                                                           corner_pixel_removed=12)

        filled = d_DE_and_filling.filling(file_handler=with_rectangle_corner,frame_ID=self.frame_ID,
                                          folder_path=self.folder_path+'\\filled',
                                          binarizing_mono=binarizing_mono,
                                          addition='_window_size_{}'.format(adaptive_window_size),
                                          write_filled=False)

        # remove added rectangle and corner
        rectangle_corner_inverse = d_DE_and_filling.draw_rectangle_corner_inverse(file_handler=filled,
                                                                                  line_thickness=12)

        # labelling and plotting
        droplet_size_value, real_area, coverage = e_labeling_and_plotting.pattern_recognition(
            file_handler=rectangle_corner_inverse,
            frame_ID=self.frame_ID,
            minimum_size=minimum_area_for_recognition,
            write_labelling=False,
            folder_path=self.folder_path+'\\labelled',
            binarizing_mono=binarizing_mono,
            addition='_window_size_{}'.format(adaptive_window_size))

        e_labeling_and_plotting.droplet_statistic(droplet_size_list=droplet_size_value,
                                                  frame_ID=self.frame_ID,
                                                  local_time=self.local_time,
                                                  coverage=coverage,
                                                  plot_statistic=False,
                                                  write_to_xlsx=False,
                                                  write_to_csv=False,
                                                  max_droplet_size_for_fixed=max_droplet_size_for_fixed,
                                                  folder_path=self.folder_path + '\\statistic',
                                                  binarizing_mono=binarizing_mono,
                                                  addition='_window_size_{}'.format(adaptive_window_size))

        return thresh

def main():
    # only on cropped images ==> image folder name: fixed_position_cropped
    file_date='01_12_2022'
    batch=''
    addition='' #'_concentration_group'
    file_date_all = ['01-12-2022', '02-12-2022']  # 'dd-mm-yyyy'
    date_string = ['Thu Dec  1', 'Fri Dec  2']
    env_start='01-12-2022 21:24:50'
    ER_start='01-12-2022 21:16:52'
    clean_all_processed_images=True
    remove_background=False
    bg_name=''
    start_idx=0
    end_idx=1

    other_pre_obj = Other_preparations(file_date=file_date, batch=batch, addition=addition)
    if remove_background:
        bg_gray=other_pre_obj.background_image_reading(background_image_name=bg_name)
    else:
        bg_gray=None

    other_pre_obj.folders_creating()
    if clean_all_processed_images:
        other_pre_obj.folders_cleaning()


    file_date_1 = file_date.split('_')
    file_date_1 = file_date_1[2] + '_' + file_date_1[1] + '_' + file_date_1[0]
    t_info_folder_path= 'E:\\images_processed\\' + file_date_1 + addition + '\\images_' + file_date + batch + '\\multiple_images'
    t_info_path=t_info_folder_path+'\\t_extracted_' + file_date + batch+ '.csv'

    if not os.path.isfile(t_info_path):
        time_reading_and_preparations_obj=Time_reading(file_date=file_date, batch=batch, addition=addition)
        time_df=time_reading_and_preparations_obj.time_procesing(file_date_all=file_date_all,
                                                                 date_string=date_string,
                                                                 year=' 2022',
                                                                 env_start=env_start,
                                                                 ER_start=ER_start)
    else:
        time_df=pd.read_csv(t_info_path, sep=',')

    for i in range(start_idx, end_idx):
        single_image_all_together_obj=Single_image_all_together(t_ID_info=time_df, i=i, file_date=file_date,
                                                                batch=batch, addition=addition)
        thresh=single_image_all_together_obj.single_image_processing(remove_background=remove_background,
                                                              gray_scale_background_image=bg_gray,
                                                              binarizing_mono=True,
                                                              adaptive_window_size=10, # specify when binarizing_mono=False
                                                              DE_method='', # DE method: OP or CL
                                                              DE_size=5, # specify when DE method is chosen
                                                              minimum_area_for_recognition=10,
                                                              max_droplet_size_for_fixed=2000, # maximum droplet value for CSV file
                                                              if_Otsu_when_mono=True, # if True, find threshold automatically with Otsu's method
                                                              threshold_when_not_Otsu_but_mono=120, # only useful when previous boolean variable is set to False
                                                              thresh_diff_for_Otsu=-8 # adding certain value in addition to Otsu's threshold. Only useful when if_Otsu_when_mono is set to True
                                                              )

        print(thresh)
if __name__ == '__main__':
    main()


















