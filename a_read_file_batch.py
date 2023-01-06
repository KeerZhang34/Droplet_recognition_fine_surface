import os
import glob
import pandas as pd

def read_ID(file_date):
    file_path='C:\\python_file\\1\\image_processing_in_batch\\time_batch\\t_extracted_'+file_date+'.csv'
    file=pd.read_csv(file_path,sep=',')


    return file

def read_figure(folder_path,frame_number):

    #all_detected_file=glob.glob('{}\\condensation_*+{}_21_10_2021.bmp'.format(folder_path.replace("'",""),frame_number))
    try:
        all_detected_file=glob.glob('{}\\*_{}_*'.format(folder_path.replace("'",""),frame_number))
        file_path_handler=all_detected_file[0]
        #length=len(all_detected_file)
        file_name=os.path.split(file_path_handler)[-1]
        return [file_path_handler, file_name]
    except IndexError:
        pass










