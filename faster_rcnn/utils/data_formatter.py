import os 
import numpy as np
import pandas as pd
import cv2
import argparse
import shutil
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import json


image_dir = ''
mask_dir = ''
phase = 'testing'

master_list = []
master_img = {
    'file_name': "",
    'height': 0,
    'width': 0,
    'image_id': "",
    'annotations': []
}

master_ann = {
    'bbox': [],
    'category_id': 0,
    'bbox_mode': 0,
}

def plot_num_classes(info_file):
    if not os.path.exists(info_file):
        raise ValueError("info_file not exist")
    else:
        num_classes_per_image = np.load(info_file)
        plt.figure(figsize=(5, 5))
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        plt.bar(class_array, num_classes_per_image)
        plt.title('Number of classes per image', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=14, fontweight='bold')
        plt.ylabel('Number of classes', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(os.path.join(os.path.dirname(info_file), 'num_classes_per_image.png'), dpi=300)



def make_folds(image_dir, mask_dir, save_dir, folds,seed=42):
    if folds <=0:
        raise ValueError("folds must be greater than 0")
    elif not os.path.exists(image_dir):
        raise ValueError("image_dir not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError("image_dir is empty")
    else:
        images = os.listdir(image_dir)
        masks = os.listdir(mask_dir)
        len_of_each_fold = len(images) // folds

        random.seed(seed)
        random.shuffle(images)

        for i in tqdm (range(folds), desc="Creating Folds...", ascii=False, ncols=75):  
            print(f"Creating fold {i+1}")
            fold_dir = os.path.join(save_dir, f"fold_{i+1}")
            im_save_dir = os.path.join(fold_dir)
            mask_save_dir = os.path.join(fold_dir)
            if not os.path.exists(im_save_dir):
                os.makedirs(im_save_dir)
            if not os.path.exists(mask_save_dir):
                os.makedirs(mask_save_dir)
            try:
                images.remove('.DS_Store')
            except:
                pass
            train_images = images[:i*len_of_each_fold] + images[(i+1)*len_of_each_fold:]
            test_images = images[i*len_of_each_fold:(i+1)*len_of_each_fold]

            for image in tqdm (train_images, desc="Creating Train...", ascii=False, ncols=75):
                im_save_path = os.path.join(im_save_dir,'train', 'images')
                mask_save_path = os.path.join(mask_save_dir,'train', 'labels')
                if not os.path.exists(im_save_path):
                    os.makedirs(im_save_path)
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                time.sleep(0.01)
                image_path = os.path.join(image_dir, image)
                mask_path = os.path.join(mask_dir, image.split('.png')[0] + '.txt')
                shutil.copy(image_path, im_save_path)
                shutil.copy(mask_path, mask_save_path)
            for image in tqdm (test_images, desc="Creating Test...", ascii=False, ncols=75):
                im_save_path = os.path.join(im_save_dir,'val', 'images')
                mask_save_path = os.path.join(mask_save_dir,'val', 'labels')
                if not os.path.exists(im_save_path):
                    os.makedirs(im_save_path)
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                time.sleep(0.01)
                image_path = os.path.join(image_dir, image)
                mask_path = os.path.join(mask_dir, image.split('.png')[0] + '.txt')
                shutil.copy(image_path, im_save_path)
                shutil.copy(mask_path, mask_save_path)



def image_info(image_dir, mask_dir, save_dir, phase):
    if not os.path.exists(image_dir):
        raise ValueError("image_dir not exist")
    elif len(os.listdir(image_dir)) == 0:
        raise ValueError("image_dir is empty")
    else:
        master_list = []
        save_dir = os.path.join(save_dir, "master")
        class_array = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
        num_classes_per_image = np.zeros(len(class_array))
        for i in tqdm (range(len(os.listdir(image_dir))), desc="Creating Master...", ascii=False, ncols=75):
            time.sleep(0.01)
            image_name = os.listdir(image_dir)[i]
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.split('.png')[0] + '.csv')
            image = cv2.imread(image_path)
            im_height, im_width, _ = image.shape
            try:
                mask = pd.read_csv(mask_path, header=0)
            except:
                # print(f"{mask_path.split('/')[-1]} not exist")
                continue
            # print(image.shape)
            # print(mask.keys())
            loc_img = master_img.copy()
            loc_img['file_name'] = image_path
            loc_img['height'] = im_height
            loc_img['width'] = im_width
            loc_img['image_id'] = image_name.split('.png')[0]
            loc_img['annotations'] = []
            for index, row in mask.iterrows():
                loc_ann = master_ann.copy()
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
                # print(f"Class: {class_name}, Class ID: {class_id}")
                loc_ann['bbox'] = [x_min, y_min, x_max, y_max]
                loc_ann['category_id'] = class_id
                loc_ann['bbox_mode'] = 0
                loc_img['annotations'].append(loc_ann.copy())
            master_list.append(loc_img.copy())
            # print(f"Image {i+1} completed")

                
            if phase == 'testing' and i == 10:
                result = "testing complete"
                # printing result in a pretty way
                print("\n")
                print("Result:")
                print("=======")
                for image in master_list:
                    print(image)
                    print("\n")
                    print("---------------------------------------------------------------------------------------------------------------------------------------")
                    print("\n")
                return result
            
        
        if phase != 'testing':
            master_list = np.array(master_list)
            np.save(os.path.join(save_dir, 'master.npy'), master_list)
            # saving as json
            object = json.dumps(master_list.tolist(), indent = 4)
            with open(os.path.join(save_dir, 'master.json'), "w+") as outfile:
                outfile.write(object)
            np.save(os.path.join(save_dir, 'num_classes_per_image.npy'), num_classes_per_image)
            print("Completed")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-m', '--mask_dir', required=True, help='mask directory')
    argparser.add_argument('-s', '--save_dir', required=True, help='save directory')
    argparser.add_argument('-f', '--folds', required=True, help='number of folds')
    argparser.add_argument('-p', '--phase', required=True, help='phase')
    argparser.add_argument('--seed', required=False, help='Seed for reproducibility')

    args = argparser.parse_args()

    image_info(args.image_dir, args.mask_dir, args.save_dir, args.phase)
    # make_folds(os.path.join (args.save_dir, 'master', 'images'), os.path.join (args.save_dir, 'master', 'labels'), args.save_dir, int(args.folds), int(args.seed))
    # plot_num_classes(os.path.join(args.save_dir,'master', 'num_classes_per_image.npy'))