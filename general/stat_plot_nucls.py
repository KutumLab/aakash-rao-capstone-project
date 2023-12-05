import os 
import numpy as np
import pandas as pd
import cv2
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import json


def plot_stats(image_dir, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    classes = ['nonTIL_stromal', 'sTIL', 'tumor_any', 'other']
    class_stats = pd.read_csv(os.path.join(image_dir, 'class_stats.csv'), header=0)
    class_stats = class_stats['0'].values.tolist()
    class_stats = np.array(class_stats)
    print(class_stats)
    num_classes_per_image = pd.read_csv(os.path.join(image_dir, 'num_classes_per_image.csv'), header=0)
    num_classes_per_image = num_classes_per_image['len_masks'].values.tolist()
    num_classes_per_image = np.array(num_classes_per_image)

    total_images = class_stats[-1]
    print("Total Images: {}".format(total_images))
    total_annotations = int(class_stats[-2])
    class_stats = class_stats[:-2].astype(int)

    plt.figure(figsize=(3,3))
    plt.bar(np.arange(len(class_stats)), class_stats)
    plt.xticks(np.arange(len(class_stats)), class_stats)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Class')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'num_images_per_class.png'), bbox_inches='tight', dpi=300)
    plt.close()


    
    

    


                



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-p', '--plot_dir', required=True, help='save directory')
    args = argparser.parse_args()

    print("Generating plots for NuCLS Dataset Statistics...")
    plot_stats(args.image_dir, args.plot_dir)