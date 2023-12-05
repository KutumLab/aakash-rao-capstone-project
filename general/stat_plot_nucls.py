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
    class_stats = pd.read_csv(os.path.join(image_dir, 'class_stats.csv'), header=0)
    num_classes_per_image = pd.read_csv(os.path.join(image_dir, 'num_classes_per_image.csv'), header=0)
    print(class_stats)
    print(num_classes_per_image)
    


                



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-p', '--plot_dir', required=True, help='save directory')
    args = argparser.parse_args()

    print("Generating plots for NuCLS Dataset Statistics...")
    plot_stats(args.image_dir, args.plot_dir)