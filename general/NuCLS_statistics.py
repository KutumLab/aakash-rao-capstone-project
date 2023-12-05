import os 
import numpy as np
import pandas as pd
import cv2
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import json


def image_info(image_dir, mask_dir, plot_dir):
    im_count = 0
    class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
    num_classes_per_image = np.zeros(len(class_array))
    if not os.path.exists(image_dir):
        raise ValueError("image_dir not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError("image_dir is empty")
    else:
        master_list = []
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        num_classes_per_image = np.zeros(len(class_array))
        for i in tqdm (range(len(os.listdir(image_dir))), desc="Creating Master...", ascii=False, ncols=75):
            time.sleep(0.01)
            image_name = os.listdir(image_dir)[i]
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.split('.png')[0] + '.csv')
            # image = cv2.imread(image_path)
            # im_height, im_width, _ = image.shape
            try:
                mask = pd.read_csv(mask_path, header=0)
            except:
                # print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            for index, row in mask.iterrows():
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                class_name = row['super_classification']
                if class_name == 'AMBIGUOUS' or class_name == 'other_nucleus':
                    class_name = 'other'
                try:
                    class_id = class_array.index(class_name)
                except ValueError:
                    raise ValueError(f"{class_name} not in class_array")
                num_classes_per_image[class_id] += 1
            im_count += 1
        print(f"Total number of images: {im_count}")
        print(f"Total number of classes: {sum(num_classes_per_image)}")
        print(f"Number of classes per image: {num_classes_per_image}")
                



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-m', '--mask_dir', required=True, help='mask directory')
    argparser.add_argument('-p', '--plot_directory', required=True, help='save directory')
    argparser.add_argument('--seed', required=False, help='Seed for reproducibility')

    args = argparser.parse_args()

    print("Generating NuCLS Dataset Statistics...")
    image_info(args.image_dir, args.mask_dir, args.plot_directory)