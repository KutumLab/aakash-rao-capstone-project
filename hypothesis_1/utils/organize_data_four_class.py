import os 
import pandas as pd
import numpy as np
import argparse
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.manifold import TSNE



def organize_and_extract(image_path, mask_path, save_path, num_classes=4):
    stromal = 0
    tumor = 0
    other = 0
    sTIL = 0
    ims_df = pd.DataFrame(columns=['image', 'label'])
    imgs_arr = []
    labels_arr = []

    for image in os.listdir(image_path):
        im_path = os.path.join(image_path, image)
        labels = os.path.join(mask_path, image[:-4] + '.csv')
        classes = ['nonTIL_stromal','sTIL','tumor_any','other']
        img = cv2.imread(im_path)
        try:
            df = pd.read_csv(labels)
            df = df.drop(['Unnamed: 0'], axis=1)
        except:
            continue

        df = df[['super_classification', 'xmax', 'xmin', 'ymax', 'ymin']]
        # rename columns
        df.columns = ['class', 'xmax', 'xmin', 'ymax', 'ymin']
        print(df.head())
        for index, row in df.iterrows():
            loc_class = row['class']
            if loc_class not in classes:
                loc_class = 'other'
            print(loc_class)
            # extract the image
            x1 = row['xmin']
            x2 = row['xmax']
            y1 = row['ymin']
            y2 = row['ymax']
            # making a square box from the rectangle
            if (x2-x1) > (y2-y1):
                y2 = y1 + (x2-x1)
            else:
                x2 = x1 + (y2-y1)
            loc_img = img[y1:y2, x1:x2]

            # loc_img = img[y1:y2, x1:x2]
            # save the image
            save_folder = os.path.join(save_path, 'master/images')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)




            if loc_class == 'nonTIL_stromal':
                save_name = os.path.join(save_folder, f'{loc_class}_{stromal}.png')
                imgs_arr.append(save_name)
                labels_arr.append(0)
                stromal += 1
            elif loc_class == 'sTIL':
                save_name = os.path.join(save_folder, f'{loc_class}_{sTIL}.png')
                imgs_arr.append(save_name)
                labels_arr.append(1)
                sTIL += 1
            elif loc_class == 'tumor_any':
                save_name = os.path.join(save_folder, f'{loc_class}_{tumor}.png')
                imgs_arr.append(save_name)
                labels_arr.append(2)
                tumor += 1
            else:
                save_name = os.path.join(save_folder, f'{loc_class}_{other}.png')
                imgs_arr.append(save_name)
                labels_arr.append(3)
                other += 1


            

            cv2.imwrite(save_name, loc_img)
    ims_df['image'] = imgs_arr
    ims_df['label'] = labels_arr
    ims_df.to_csv(os.path.join(save_path, 'master/master.csv'), index=False)  
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize and extract data from the raw data')
    parser.add_argument('-i', '--image_path', type=str, help='Path to the image folder')
    parser.add_argument('-m', '--mask_path', type=str, help='Path to the mask folder')
    parser.add_argument('-s', '--save_path', type=str, help='Path to save the extracted data')
    args = parser.parse_args()
    organize_and_extract(args.image_path, args.mask_path, args.save_path)
    