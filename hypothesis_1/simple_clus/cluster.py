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


def gen_scatter(path):
    im_path = os.path.join(path, "datasets/clus/master/images")
    csv_save_path = os.path.join(path, "output/csvs")
    plot_save_path = os.path.join(path, "output/plots")
    npy_save_path = os.path.join(path, "output/npys")
    if not os.path.exists(os.path.join(csv_save_path,'tsne.csv')):
        X = None
        labels = pd.read_csv(os.path.join(path, 'datasets/clus/master/master.csv'))
        labels = labels['label'].values
        for image in tqdm(os.listdir(im_path)):
            im = cv2.imread(os.path.join(im_path, image))
            im = cv2.resize(im, (50,50))
            im = im.reshape(1, -1)
            if X is None:
                X = im
            else:
                X = np.concatenate((X, im), axis=0)
        X = X.reshape(-1, 50*50*3)
        print(X.shape)
        np.save(os.path.join(npy_save_path,'dataset_all.npy'), X)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(X).T
        print(tsne_results[0].shape)
        
        df_tsne = pd.DataFrame({'TSNE-1':tsne_results[0], 'TSNE-2':tsne_results[1], 'class':labels})
        df_tsne.to_csv(os.path.join(csv_save_path,'tsne.csv'), index=False)
        print(df_tsne.head())
    else:
        print('Loading t-sne results from csv')
        df_tsne = pd.read_csv(os.path.join(csv_save_path,'tsne.csv'))
    
    df_tsne['TSNE-1'] = df_tsne['TSNE-1']
    df_tsne['TSNE-2'] = df_tsne['TSNE-2']
    plt.figure(figsize=(4,4))
    cls_1 = df_tsne[df_tsne['class'] == 0]
    cls_2 = df_tsne[df_tsne['class'] == 1]
    cls_3 = df_tsne[df_tsne['class'] == 2]
    cls_4 = df_tsne[df_tsne['class'] == 3]
    cls_1 = plt.scatter(cls_1['TSNE-1'], cls_1['TSNE-2'], c='#D7263D', label='Stromal',marker='o', s=0.1)
    cls_2 = plt.scatter(cls_2['TSNE-1'], cls_2['TSNE-2'], c='#F46036', label='sTIL',marker='o', s=0.1)
    cls_3 = plt.scatter(cls_3['TSNE-1'], cls_3['TSNE-2'], c='#2E294E', label='Tumor',marker='o', s=0.1)
    cls_4 = plt.scatter(cls_4['TSNE-1'], cls_4['TSNE-2'], c='#1B998B', label='Other',marker='o', s=0.1)

    plt.legend(handles=[cls_1, cls_2, cls_3, cls_4], loc='upper right', fontsize=6)
    plt.title('NuCLS Dataset (T-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('1st component', fontsize=14, fontweight='bold')
    plt.ylabel('2nd component', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_path, 'nuCLS_TSNE_plot.png'), dpi=300)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize and extract data from the raw data')
    parser.add_argument('-s', '--save_path', type=str, help='Path to save the extracted data')
    args = parser.parse_args()
    gen_scatter(args.save_path)