import numpy as np
import os
from skimage.measure import label,regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import pandas as pd
import xlsxwriter

def pattern_recognition(file_handler,frame_ID,minimum_size,write_labelling=True, folder_path='', binarizing_mono=True, addition=''):
    # minimum_size: thershold of pattern size in pixel under which the pattern will be ignored

    h,w=file_handler.shape[:2]

    fig, ax=plt.subplots(figsize=(20,15))
    plt.xticks([])
    plt.yticks([])
    ax.imshow(file_handler,cmap='gray')

    labels=label(file_handler)

    droplet_size_value = []
    real_area = 0

    for region in regionprops(labels):
        if region.area<=minimum_size:
            continue
        else:
            minr, minc, maxr, maxc =region.bbox
            rect = mpatches.Rectangle((minc,minr),maxc-minc,maxr-minr, fill='False', facecolor='None', edgecolor='red', linewidth=0.7)
            ax.add_patch(rect)

            diameter = 2 * math.sqrt(region.area / math.pi)
            real_area += region.area
            droplet_size_value.append(diameter)

    coverage='{}%'.format('{:.2f}'.format(real_area*100/(h*w)))

    if write_labelling:

        os.chdir(folder_path)
        plt.savefig('{}_labelled_mono.png'.format(frame_ID)) if binarizing_mono \
            else plt.savefig('{}_labelled_adaptive{}.png'.format(frame_ID,addition))




    plt.clf()
    plt.close(fig)

    return droplet_size_value,real_area,coverage

def count_droplet(droplet_size_list,steps,min_list,max_list):

    # min_list, max_list and droplet_size_list: in pixel
    pixel_size = 3.45 / 10
    calculated_area = 0
    list_interval = float((max_list - min_list) / steps)

    list_value = list(np.zeros(steps))
    list_range = list(np.zeros(steps))
    num_droplets=0

    for k in range(steps):

        mid_value = "{:.2f}".format((min_list + (k + 0.5) * list_interval) * pixel_size)
        float_mid_value = float(mid_value)
        list_range[k] = float_mid_value

        for d in droplet_size_list:
            if d >= min_list + k * list_interval and d < min_list + (k + 1) * list_interval:
                list_value[k] += 1
                calculated_area += math.pi * (min_list + (k + 0.5) * list_interval) * (
                        min_list + (k + 0.5) * list_interval) / 4
                num_droplets+=1
            else:
                continue

    if num_droplets < len(droplet_size_list):
        return None

    return [list_range,list_value,num_droplets]

class write_CSV:
    def __init__(self, list_range, folder_path='', binarizing_mono=True, addition=''):
        file_path = folder_path + '_droplet_statistics_mono.csv' if binarizing_mono else \
            folder_path + '_droplet_statistics_adaptive{].csv'.format(addition)
        self.file_path = file_path
        if not os.path.isfile(file_path):
            df = {'Droplet size': list_range}
            dataFrame = pd.DataFrame(data=df)
            dataFrame.to_csv(file_path, sep=',', index=False)
    def add_column(self,list_value,local_time):

        file=pd.read_csv(self.file_path,sep=',')
        file['{}'.format(local_time)] = list_value #self=list_value
        file.to_csv(self.file_path, sep=',', index=False)



def droplet_statistic(droplet_size_list,frame_ID,local_time,coverage,plot_statistic=True,write_to_xlsx=True,
                      write_to_csv=False, max_droplet_size_for_fixed=2000, folder_path='', binarizing_mono=True, addition=''):

    if write_to_xlsx:

        min_list = min(droplet_size_list)
        max_list = max(droplet_size_list) + 1

        #list_range and list_value: in pixel
        list_range,list_value,num_droplets=count_droplet(droplet_size_list,steps=100,min_list=min_list,max_list=max_list)


        if plot_statistic:
            fig, ax = plt.subplots()
            ax.set_title('Statistics')
            ax.bar(list_range, list_value)
            ax.set_xlabel('(Equivalent) diameter, average of each diameter range (microns)')
            ax.set_ylabel('Amount')

            os.chdir(folder_path)
            plt.savefig('{}_statistic_adaptive_range_mono_thresh.png'.format(frame_ID)) if binarizing_mono else \
                plt.savefig('{}_statistic_adaptive_range_adaptive_thresh{}.png'.format(frame_ID, addition))
            plt.clf()
            plt.close(fig)


        xlsx_path=folder_path+'\\statistic_data_mono_thresh.xlsx' if binarizing_mono else \
            folder_path+'\\statistic_data_adaptive_thresh{}.xlsx'.format(addition)
        with xlsxwriter.Workbook(xlsx_path) as workbook:

            count = []
            count.append(list_range)
            count.append(list_value)
            worksheet = workbook.add_worksheet(name='ID={}'.format(frame_ID))

            bold = workbook.add_format({'bold': True})
            worksheet.write('A1', 'Local time', bold)
            worksheet.write('B1', local_time)
            worksheet.write('A2', 'Surface coverage', bold)
            worksheet.write('B2', coverage)
            worksheet.write('A3', 'Droplet size', bold)
            worksheet.write('B3', 'Droplet amount', bold)
            for col_num, data in enumerate(count):
                worksheet.write_column(3, col_num, data)

    if write_to_csv:
        try:
            min_list=0
            max_list=max_droplet_size_for_fixed
            list_range, list_value, num_droplets = count_droplet(droplet_size_list, steps=200, min_list=min_list, max_list=max_list)
            list_range.append('Surface coverage')
            list_value.append(coverage)
            list_range.append('Droplet amount')
            list_value.append(num_droplets)


            if plot_statistic:
                fig, ax = plt.subplots()
                ax.set_title('Statistics')
                ax.bar(list_range, list_value)
                ax.set_xlabel('(Equivalent) diameter, average of each diameter range (microns)')
                ax.set_ylabel('Amount')

                os.chdir(folder_path)
                plt.savefig('{}_statistic_fixed_range_mono_thresh.png'.format(frame_ID)) if binarizing_mono else \
                    plt.savefig('{}_statistic_fixed_range_adaptive_thresh{}.png'.format(frame_ID, addition))
                plt.clf()
                plt.close(fig)

            CSV_file=write_CSV(list_range=list_range, folder_path=folder_path, binarizing_mono=binarizing_mono, addition=addition)
            CSV_file.add_column(list_value=list_value,local_time=local_time)
            

        except ValueError:
            raise ValueError("Droplets diameter out of range. Check filling procedure, and change max_value_for_fixed if really needed. By default that value equals to 2000.")

        except IndexError:
            raise IndexError("End of figure reading.")

        except TypeError:
            raise TypeError("max_list_value_for_fixed too small, change to a bigger one. Or bad thresholding, resulting in unexpected big blanks.")


















