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




def organize_and_extract(image_path, mask_path, save_path):
    stromal = 0
    tumor = 0
    other = 0
    sTIL = 0
    ims_df = pd.DataFrame(columns=['image', 'label'])
    imgs_arr = []
    labels_arr = []
    metrics = pd.DataFrame(columns=['box_height', 'box_width', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'median_r', 'median_g', 'median_b', 'pixel_count', 'class'])
    box_height = []
    box_width = []
    mean_r = []
    mean_g = []
    mean_b = []
    std_r = []
    std_g = []
    std_b = []
    median_r = []
    median_g = []
    median_b = []
    pixel_count = []

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


            # extract the metrics
            box_height.append(y2-y1)
            box_width.append(x2-x1)
            mean_r.append(np.mean(loc_img[:,:,0]))
            mean_g.append(np.mean(loc_img[:,:,1]))
            mean_b.append(np.mean(loc_img[:,:,2]))
            std_r.append(np.std(loc_img[:,:,0]))
            std_g.append(np.std(loc_img[:,:,1]))
            std_b.append(np.std(loc_img[:,:,2]))
            median_r.append(np.median(loc_img[:,:,0]))
            median_g.append(np.median(loc_img[:,:,1]))
            median_b.append(np.median(loc_img[:,:,2]))
            pixel_count.append((y2-y1)*(x2-x1))

            cv2.imwrite(save_name, loc_img)
    ims_df['image'] = imgs_arr
    ims_df['label'] = labels_arr
    ims_df.to_csv(os.path.join(save_path, 'master/master.csv'), index=False)  

    metrics['box_height'] = box_height
    metrics['box_width'] = box_width
    metrics['mean_r'] = mean_r
    metrics['mean_g'] = mean_g
    metrics['mean_b'] = mean_b
    metrics['std_r'] = std_r
    metrics['std_g'] = std_g
    metrics['std_b'] = std_b
    metrics['median_r'] = median_r
    metrics['median_g'] = median_g
    metrics['median_b'] = median_b
    metrics['pixel_count'] = pixel_count
    metrics['class'] = labels_arr
    metrics.to_csv(os.path.join(save_path, 'master/metrics.csv'), index=False)
    pass

def plot_few(path):
    csv_path = os.path.join(path, 'master/master.csv')
    classes = ['nonTIL_stromal','sTIL','tumor_any','other']
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df['label'].value_counts())
    df_0 = df[df['label'] == 0]
    df_0 = df_0.sample(n=10)
    df_1 = df[df['label'] == 1]
    df_1 = df_1.sample(n=10)
    df_2 = df[df['label'] == 2]
    df_2 = df_2.sample(n=10)
    df_3 = df[df['label'] == 3]
    df_3 = df_3.sample(n=10)
    plt.figure(figsize=(7,7))
    for i in range(10):
        plt.subplot(4, 10, i+1)
        image = cv2.imread(df_0.iloc[i]['image'])
        image = cv2.resize(image, (100,100))
        plt.imshow(image)
        plt.axis('off')
        plt.title('Stromal')
    for i in range(10):
        plt.subplot(4, 10, i+11)
        image = cv2.imread(df_1.iloc[i]['image'])
        image = cv2.resize(image, (100,100))
        plt.imshow(image)
        plt.axis('off')
        plt.title('sTIL')
    for i in range(10):
        plt.subplot(4, 10, i+21)
        image = cv2.imread(df_2.iloc[i]['image'])
        image = cv2.resize(image, (100,100))
        plt.imshow(image)
        plt.axis('off')
        plt.title('Tumor')
    for i in range(10):
        plt.subplot(4, 10, i+31)
        image = cv2.imread(df_3.iloc[i]['image'])
        image = cv2.resize(image, (100,100))
        plt.imshow(image)
        plt.axis('off')
        plt.title('Other')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(path, 'master/sample_plot.png'), dpi=300)

    pass


def genpca(path):
    im_path = os.path.join(path, 'master/images')
    if not os.path.join(path, 'master/pca_plot.png'):
        X = None
        test_shape = len(os.listdir(im_path))
        labs = pd.read_csv(os.path.join(path, 'master/metrics.csv'))['class'].values[:test_shape]

        for image in tqdm(os.listdir(im_path)[:test_shape]):
            im = cv2.imread(os.path.join(im_path, image))
            im = cv2.resize(im, (50,50))
            im = im.reshape(1, -1)
            if X is None:
                X = im
            else:
                X = np.concatenate((X, im), axis=0)
        X = X.reshape(-1, 50*50*3).T
        print(X.shape)
        pca = PCA(n_components=2)
        pca.fit_transform(X)
        print(pca.components_.shape)
        df_pca = pd.DataFrame({'pca1':pca.components_[0], 'pca2':pca.components_[1], 'class':labs})
        df_pca.to_csv(os.path.join(path, 'master/pca.csv'), index=False)
        print(df_pca.head())
    else:
        df_pca = pd.read_csv(os.path.join(path, 'master/pca.csv'))
    
    df_pca['pca1'] = df_pca['pca1']*100
    df_pca['pca2'] = df_pca['pca2']*100
    plt.figure(figsize=(4,4))
    cls_1 = df_pca[df_pca['class'] == 0]
    cls_2 = df_pca[df_pca['class'] == 1]
    cls_3 = df_pca[df_pca['class'] == 2]
    cls_4 = df_pca[df_pca['class'] == 3]
    cls_1 = plt.scatter(cls_1['pca1'], cls_1['pca2'], c='#D7263D', label='Stromal',marker='o', s=0.1)
    cls_2 = plt.scatter(cls_2['pca1'], cls_2['pca2'], c='#F46036', label='sTIL',marker='o', s=0.1)
    cls_3 = plt.scatter(cls_3['pca1'], cls_3['pca2'], c='#2E294E', label='Tumor',marker='o', s=0.1)
    cls_4 = plt.scatter(cls_4['pca1'], cls_4['pca2'], c='#1B998B', label='Other',marker='o', s=0.1)

    plt.legend(handles=[cls_1, cls_2, cls_3, cls_4], loc='upper right', fontsize=8)
    plt.title('PCA plot of the data', fontsize=14, fontweight='bold')
    plt.xlabel('PCA1', fontsize=14, fontweight='bold')
    plt.ylabel('PCA2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'master/pca_plot.png'), dpi=300)
    # plt.show()




    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize and extract data from the raw data')
    parser.add_argument('-i', '--image_path', type=str, help='Path to the image folder')
    parser.add_argument('-m', '--mask_path', type=str, help='Path to the mask folder')
    parser.add_argument('-s', '--save_path', type=str, help='Path to save the extracted data')
    args = parser.parse_args()
    # organize_and_extract(args.image_path, args.mask_path, args.save_path)
    # plot_few(args.save_path)
    genpca(args.save_path)