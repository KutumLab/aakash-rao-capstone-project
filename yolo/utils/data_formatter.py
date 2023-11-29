import os 
import numpy
import pandas as pd
import cv2
import argparse


image_dir = ''
mask_dir = ''
phase = 'testing'

def image_info(image_dir, mask_dir, save_dir, phase):
    if not os.path.exists(image_dir):
        raise ValueError("image_dir not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError("image_dir is empty")
    else:
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.split('.png')[0] + '.csv')
            image = cv2.imread(image_path)
            mask = pd.read_csv(mask_path, header=0)
            print(image.shape)
            print(mask.keys())
            for index, row in mask.iterrows():
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                class_name = row['super_classification']
                class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other_nucleus']
                if class_name == 'AMBIGUOUS' or class_name == 'other_nucleus':
                    class_name = 'other'
                try:
                    class_id = class_array.index(class_name)
                except ValueError:
                    raise ValueError("class_name not in class_array")
                
                print(x_min, y_min, x_max, y_max, class_name, class_id)

            if phase == 'testing':
                result = "testing complete"
                return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-m', '--mask_dir', required=True, help='mask directory')
    argparser.add_argument('-s', '--save_dir', required=True, help='save directory')
    argparser.add_argument('-p', '--phase', required=True, help='phase')

    args = argparser.parse_args()

    result = image_info(args.image_dir, args.mask_dir, args.save_dir, args.phase)
    print(result)