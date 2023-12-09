#!/usr/bin/env python
# coding: utf-8

#Importing necessary packages
import pandas as pd
import sys
from sklearn.decomposition import PCA
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
import shutil

from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
def train(model_type, epochs, batch_size,learning_rate, savepath):
    #checking tensorflow version
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train_large = '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/images'

    #(trainX, testX, trainY, testY) = train_test_split(data, train_large.target, test_size=0.25)

    # ImageDataGenerator
    # color images
    print(f'Model Type: {model_type}')
    dataframe = pd.read_csv('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/master.csv')
    clases = ['nonTIL_stromal','sTIL','tumor_any','other']
    # map df to classes
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any',3:'other'})
    test_df = dataframe.sample(frac=0.2, random_state=42)
    dataframe = dataframe.drop(test_df.index)
    print(dataframe.head())

    datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,vertical_flip=True)
    # Training Data with Augmentation
    train_generator = datagen_train.flow_from_dataframe(
        dataframe=dataframe,
        directory=train_large,
        x_col="image",
        y_col="label",
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(50,50))
    #Validation Data
    valid_generator = datagen_train.flow_from_dataframe(
        dataframe=dataframe,
        directory=train_large,
        x_col="image",
        y_col="label",
        subset="validation",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(50,50))

    # finding class keys
    class_keys = train_generator.class_indices.keys()
    print(class_keys)
    class_keys = valid_generator.class_indices.keys()
    print(class_keys)


    # if model_type == 'InceptionV3':
    # 	inception = InceptionV3(
    # 			weights='imagenet',
    # 			include_top=False,
    # 			input_shape=(300,300,3)
    # 			)
    # 	for layer in inception.layers:
    # 			layer.trainable = True
    # 	x = layers.Flatten()(inception.output)
    # 	# adding avgpool layer
    # 	x = layers.GlobalAveragePooling2D()(inception.output)
    # 	x = layers.Dense(1024, activation = 'relu', kernel_regularizer=l2(0.01))(x)
    # 	x = layers.Dense(3, activation = 'softmax', kernel_regularizer=l2(0.01))(x)
    # 	model = Model(inception.input, x)
    # 	model.compile(optimizer = RMSprop(learning_rate = 0.0000001), loss = 'categorical_crossentropy', metrics = ['acc'])

    # Creating the model
    if model_type == 'InceptionV3':
            inception = InceptionV3(
                    weights=None,
                    include_top=False,
                    input_shape=(300,300,3)
                    )
            for layer in inception.layers:
                    layer.trainable = True
            # x = layers.Flatten()(inception.output)
            x = layers.GlobalAveragePooling2D()(inception.output)
            x = layers.Dense(128, activation = 'relu', kernel_regularizer=l2(0.01))(x)
            x = layers.Dense(4, activation = 'softmax',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            model = Model(inception.input, x)
            model.compile(optimizer = RMSprop(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])
    if model_type == 'Xception':
            xception = Xception(
                    weights=None,
                    include_top=False,
                    input_shape=(300,300,3)
                    )
            for layer in xception.layers:
                    layer.trainable = True
            # x = layers.Flatten()(xception.output)
            x = layers.GlobalAveragePooling2D()(xception.output)
            x = layers.Dense(128, activation = 'relu', kernel_regularizer=l2(0.01))(x)
            x = layers.Dense(4, activation = 'softmax',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            model = Model(xception.input, x)
            model.compile(optimizer = RMSprop(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    if not os.path.exists(f'{savepath}/{model_type}'):
        os.makedirs(f'{savepath}/{model_type}')


    #TF_CPP_MIN_LOG_LEVEL=2
    # Training the model

    print("------------------------------------------")
    print(f'Training the model {model_type}')
    print("------------------------------------------")
    filepath = f'{savepath}/{model_type}/model_log'
    if os.path.exists(filepath):
            os.makedirs(filepath)
    filepath = filepath + "/model-{epoch:02d}-{val_acc:.2f}.h5"
    callbacks = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=20)
    history = model.fit(train_generator, validation_data = valid_generator, verbose=1, epochs=epochs, callbacks=callbacks)

    print("------------------------------------------")
    print(f'Training Complete')
    print("------------------------------------------")
    # Creating a directory to save the model paths 

    # Saving the model
    model.save(f'{savepath}/{model_type}/{model_type}.h5')
    print("------------------------------------------")
    print(f'Model saved')
    print("------------------------------------------")


    #plotting the accuracy and loss
    print("------------------------------------------")
    print(f'Plotting and supplimentary data')
    print("------------------------------------------")
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{savepath}/{model_type}/Accuracy.png')

    #np.save('{savepath}/{model_type}/history1.npy',history.history)

    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = f'{savepath}/{model_type}/history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv: 
    hist_csv_file = f'{savepath}/{model_type}/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        
    loaded_model = load_model(f'{savepath}/{model_type}/{model_type}.h5')
    outcomes = loaded_model.predict(valid_generator)
    y_pred = np.argmax(outcomes, axis=1)
    # confusion matrix
    confusion = confusion_matrix(valid_generator.classes, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{savepath}/{model_type}/Confusion_matrix.png')


    conf_df = pd.DataFrame(confusion, index = class_keys, columns = class_keys)
    conf_df.to_csv(f'{savepath}/{model_type}/Confusion_matrix.csv')

    # classification report
    target_names = class_keys
    report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'{savepath}/{model_type}/Classification_report.csv')

    print("------------------------------------------")
    print(f'Supplimentary Data Saved')
    print("------------------------------------------")


    print("------------------------------------------")
    print(f'Model Evaluation')
    print("------------------------------------------")

    # Evaluating the model
    model = load_model(f'{savepath}/{model_type}/{model_type}.h5')
    test_data = ImageDataGenerator(rescale = 1.0/255.0)
    test_generator = test_data.flow_from_dataframe(
        dataframe=test_df,
        directory=train_large,
        x_col="image",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(50,50))
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc}')

    pred = model.predict(test_generator)
    pred_df = pd.DataFrame(pred)
    pred_df.to_csv(f'{savepath}/{model_type}/Predictions.csv')

    y_truth = test_generator.classes
    y_pred = np.argmax(pred, axis=1)
    confusion = confusion_matrix(y_truth, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{savepath}/{model_type}/Confusion_matrix_test.png')

    conf_df = pd.DataFrame(confusion, index = class_keys, columns = class_keys)
    conf_df.to_csv(f'{savepath}/{model_type}/Confusion_matrix_test.csv')


def test_best(savepath, model_type, batch_size):
    train_large = '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/images'
    dataframe = pd.read_csv('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/master.csv')
    clases = ['nonTIL_stromal','sTIL','tumor_any','other']
    # map df to classes
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any',3:'other'})
    test_df = dataframe.sample(frac=0.2, random_state=42)
    test_df.to_csv(os.path.join(savepath, model_type, 'test.csv'))
    outpath = os.path.join(savepath, model_type)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # Evaluating the model
    model_dir = os.path.join(savepath, model_type,"model_log")
    model = load_model(f'{model_dir}/model-300-0.70.h5')
    test_data = ImageDataGenerator(rescale = 1.0/255.0)
    test_generator = test_data.flow_from_dataframe(
        dataframe=test_df,
        directory=train_large,
        x_col="image",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(50,50))
    

    # acc, loss = model.evaluate(test_generator, verbose=1)
    # print(f'Test Loss: {loss}')
    # print(f'Test Accuracy: {acc}')
    class_keys = test_generator.class_indices.keys()
    gt_labels = test_generator.classes
    pred = model.predict(test_generator)
    pred_df = pd.DataFrame(pred)
    test_pred = pred_df 
    pred_cls = []
    for indes, row in test_pred.iterrows():
        pred_cls.append(row.values.argmax())
    test_pred['gt'] =gt_labels
    test_pred['pred'] = pred_cls

    test_pred.columns = ['nonTIL_stromal','sTIL','tumor_any','other','gt','pred']
    print(test_pred.head())
    print(len(test_pred[test_pred['gt']==test_pred['pred']])/test_pred.shape[0])
    test_pred.to_csv(f'{outpath}/gt_pred.csv')
    
    

def gen_stage_wise_conf(savepath, model_type, batch_size):
    train_large = '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/images'
    dataframe = pd.read_csv('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/master.csv')
    clases = ['nonTIL_stromal','sTIL','tumor_any','other']
    # map df to classes
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any',3:'other'})
    test_df = dataframe.sample(frac=0.2, random_state=42)
    test_df.to_csv(os.path.join(savepath, model_type, 'test.csv'))
    outpath = os.path.join(savepath, model_type, "epoch_wise_conf")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # Evaluating the model
    model_dir = os.path.join(savepath, model_type,"model_log")
    overall_acc = []
    overal_loss = []
    model_epochs = []
    for epoch_model in os.listdir(model_dir):
        model = load_model(f'{model_dir}/{epoch_model}')
        test_data = ImageDataGenerator(rescale = 1.0/255.0)
        test_generator = test_data.flow_from_dataframe(
            dataframe=test_df,
            directory=train_large,
            x_col="image",
            y_col="label",
            batch_size=batch_size,
            seed=42,
            shuffle=False,
            class_mode="categorical",
            target_size=(50,50))
        class_keys = test_generator.class_indices.keys()
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_acc}')
        overall_acc.append(test_acc)
        overal_loss.append(test_loss)
        model_epochs.append(int(epoch_model.split('-')[1]))
        print(overall_acc, overal_loss, model_epochs)

        pred = model.predict(test_generator)
        pred_df = pd.DataFrame(pred)
        pred_df.to_csv(f'{outpath}/pred_{epoch_model[:-3]}.csv')

        y_truth = test_generator.classes
        y_pred = np.argmax(pred, axis=1)
        confusion = confusion_matrix(y_truth, y_pred)

        conf_df = pd.DataFrame(confusion, index = class_keys, columns = class_keys)
        conf_df.to_csv(f'{outpath}/cm_{epoch_model[:-3]}.csv')
    df = pd.DataFrame({'epoch':model_epochs, 'acc':overall_acc, 'loss':overal_loss})
    df.to_csv(os.path.join(savepath,model_type, 'epoch_wise_acc.csv'))


def plot_model_acc(path, model):
    line_width = 0.85
    font_size = 14
    path = os.path.join(path, model)
    plot_path = os.path.join(path, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    history = pd.read_csv(os.path.join(path, 'history.csv'))
    history.columns = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
    epoch_wise_acc = pd.read_csv(os.path.join(path, 'epoch_wise_acc.csv'))
    print(history.head())
    plt.figure(figsize=(3, 3))
    plt.locator_params(nbins=5)
    plt.plot(history['acc'], label='Training Accuracy', color='blue', linewidth=line_width)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='red', linewidth=line_width)
    plt.scatter(epoch_wise_acc['epoch'], epoch_wise_acc['acc'], color='green', s=3)
    plt.title('Train-time Accuracy', fontsize=font_size, fontweight='bold')
    plt.xlabel('Epoch', fontsize=font_size, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=font_size, fontweight='bold')
    plt.ylim(0, 1)
    plt.xlim(0, 300)    
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(['Train', 'Validation','Test'], loc='best', fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'Accuracy.png'), dpi=300)
    plt.close
    pass

def plot_model_loss(path, model):
    line_width = 0.85
    font_size = 14
    path = os.path.join(path, model)
    plot_path = os.path.join(path, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    history = pd.read_csv(os.path.join(path, 'history.csv'))
    history.columns = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
    epoch_wise_acc = pd.read_csv(os.path.join(path, 'epoch_wise_acc.csv'))
    print(history.head())
    plt.figure(figsize=(3, 3))
    plt.locator_params(nbins=5)
    plt.plot(history['loss'], label='Training Loss', color='blue', linewidth=line_width)
    plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=line_width)
    plt.scatter(epoch_wise_acc['epoch'], epoch_wise_acc['loss'], color='green', s=3)
    plt.title('Train-time Loss', fontsize=font_size, fontweight='bold')
    plt.xlabel('Epoch', fontsize=font_size, fontweight='bold')
    plt.ylabel('Loss', fontsize=font_size, fontweight='bold')
    plt.ylim(2.5, 4)
    plt.xlim(0, 300)    
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(['Train', 'Validation','Test'], loc='best', fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'Loss.png'), dpi=300)
    plt.close
    pass

def plot_model_conf(path, model):
    path = os.path.join(path, model)
    plot_path = os.path.join(path, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    matrix = pd.read_csv(os.path.join(path, 'Confusion_matrix_test.csv'))
    classes = ['Stromal','sTIL','Tumor','Other']
    matrix = matrix.drop(columns=['Unnamed: 0'])
    matrix.columns = classes
    matrix.index = classes
    for col in classes:
        matrix[col] = round(matrix[col]/matrix[col].sum()*100,2)
    print(matrix.head())
    plt.figure(figsize=(4,4))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Ground Truth Labels', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Labels', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # custom legend
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(plot_path,'confusion.png'), dpi=300)
    plt.close()
    pass


def precision_recall(path):
    line_width = 0.85
    test_pred = pd.read_csv(os.path.join(path, 'gt_pred.csv'))
    test_pred = test_pred.drop(columns=['Unnamed: 0'])
    classes = ['Stromal','sTIL','Tumor','Other']
    test_pred.columns = [0,1,2,3,'gt','pred']
    plotpath = os.path.join(path, 'plots')
    print(test_pred.head())

    # generating precision recall curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize
    
    # Binarize the output
    y = label_binarize(test_pred['gt'], classes=[0, 1, 2, 3])
    n_classes = y.shape[1]
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i],
                                                            test_pred[i])
        average_precision[i] = average_precision_score(y[:, i], test_pred[i])

    # Plot Precision-Recall curve
    plt.figure(figsize=(5,4))
    plt.plot([0, 1], [1, 0], 'k--', lw=line_width, alpha=0.5)
    plt.step(recall[0], precision[0], where='post', label='Stromal', linewidth=line_width)
    plt.step(recall[1], precision[1], where='post', label='sTIL', linewidth=line_width)
    plt.step(recall[2], precision[2], where='post', label='Tumor', linewidth=line_width)
    plt.step(recall[3], precision[3], where='post', label='Other', linewidth=line_width)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plotpath,'precision_recall.png'), dpi=300)
    # plt.show()
    plt.close()
    pass


def roc_auc(path):
    line_width = 0.85
    test_pred = pd.read_csv(os.path.join(path, 'gt_pred.csv'))
    test_pred = test_pred.drop(columns=['Unnamed: 0'])
    classes = ['Stromal','sTIL','Tumor','Other']
    test_pred.columns = [0,1,2,3,'gt','pred']
    plotpath = os.path.join(path, 'plots')
    print(test_pred.head())

    # generating precision recall curve
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y = label_binarize(test_pred['gt'], classes=[0, 1, 2, 3])
    n_classes = y.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], test_pred[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(4,4))
    plt.plot([0, 1], [0, 1], 'k--', lw=line_width, alpha=0.5)
    plt.plot(fpr[0], tpr[0], label='Stromal', linewidth=line_width)
    plt.plot(fpr[1], tpr[1], label='sTIL', linewidth=line_width)
    plt.plot(fpr[2], tpr[2], label='Tumor', linewidth=line_width)
    plt.plot(fpr[3], tpr[3], label='Other', linewidth=line_width)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plotpath,'roc_auc.png'), dpi=300)
    # plt.show()
    plt.close()
    pass


def feature_gen(savepath, model_type, batch_size):
    train_large = '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/images'
    dataframe = pd.read_csv('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/master.csv')
    clases = ['nonTIL_stromal','sTIL','tumor_any','other']
    # map df to classes
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any',3:'other'})
    test_df = dataframe.sample(frac=0.2, random_state=42)
    test_df.to_csv(os.path.join(savepath, model_type, 'test.csv'))
    outpath = os.path.join(savepath, model_type)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # Evaluating the model
    model_dir = os.path.join(savepath, model_type,"model_log")
    model = load_model(f'{model_dir}/model-300-0.70.h5')
    # using only till the global_average_pooling2d layer
    model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
    model.summary()
    test_data = ImageDataGenerator(rescale = 1.0/255.0)
    test_generator = test_data.flow_from_dataframe(
        dataframe=test_df,
        directory=train_large,
        x_col="image",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(50,50))
    # getting the features
    features = model.predict(test_generator)

    class_1_indices = np.where(np.array(test_generator.classes)==0)[0]
    class_2_indices = np.where(np.array(test_generator.classes)==1)[0]
    class_3_indices = np.where(np.array(test_generator.classes)==2)[0]
    class_4_indices = np.where(np.array(test_generator.classes)==3)[0]
    print(class_1_indices.shape)
    print(class_2_indices.shape)
    print(class_3_indices.shape)
    print(class_4_indices.shape)

    # tsne
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    print(tsne_results.shape)

    # plotting
    plt.figure(figsize=(4,4))
    cls_1 = plt.scatter(tsne_results[class_1_indices,0], tsne_results[class_1_indices,1], c='#D7263D', label='Stromal',marker='o', s=0.1)
    cls_2 = plt.scatter(tsne_results[class_2_indices,0], tsne_results[class_2_indices,1], c='#F46036', label='sTIL',marker='o', s=0.1)
    cls_3 = plt.scatter(tsne_results[class_3_indices,0], tsne_results[class_3_indices,1], c='#2E294E', label='Tumor',marker='o', s=0.1)
    cls_4 = plt.scatter(tsne_results[class_4_indices,0], tsne_results[class_4_indices,1], c='#1B998B', label='Other',marker='o', s=0.1)

    plt.legend(handles=[cls_1, cls_2, cls_3, cls_4], loc='upper right', fontsize=6)
    plt.title('Xception Features from\nPooling Layer (T-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('1st component', fontsize=14, fontweight='bold')
    plt.ylabel('2nd component', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outpath,"plots",'tsne.png'), dpi=300)
    plt.close()




def gen_few_preds(savepath, model_type, batch_size):
    train_large = '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/images'
    dataframe = pd.read_csv(f'{savepath}/{model_type}/gt_pred.csv')
    correct_df = dataframe[dataframe['gt']==dataframe['pred']]
    dataframe = dataframe[dataframe['pred']!=dataframe['gt']]
    required_indices = dataframe.index.tolist()
    test_dataframe = pd.read_csv('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/master.csv')
    # map df to classes
    test_dataframe['label'] = test_dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any',3:'other'})
    test_dataframe = test_dataframe.sample(frac=0.2, random_state=42)
    test_data = ImageDataGenerator(rescale = 1.0/255.0)
    test_generator = test_data.flow_from_dataframe(
        dataframe=test_dataframe,
        directory=train_large,
        x_col="image",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(50,50))
    image_names = test_generator.filenames
    class_mapping = test_generator.classes
    # filtering required indices
    image_names = [image_names[ind] for ind in required_indices]
    class_names = [class_mapping[key] for key in required_indices]

    
    weirdo_df = pd.DataFrame(columns=['filename', 'label', 'pred'])
    weirdo_df['filename'] = image_names
    weirdo_df['label'] = class_names
    weirdo_df['pred'] = dataframe['pred'].values

    pred_mapping = test_generator.class_indices
    pred_mapping = {v: k for k, v in pred_mapping.items()}
    weirdo_df['label'] = weirdo_df['label'].map(pred_mapping)
    weirdo_df['pred'] = weirdo_df['pred'].map(pred_mapping)


    
    for class_name in weirdo_df['label'].unique():
        temp = weirdo_df.iloc[np.where(weirdo_df['label']==class_name)[0]]
        for class_name2 in temp['pred'].unique():
            temp2 = temp.iloc[np.where(temp['pred']==class_name2)[0]]
            new_file_name = f'{class_name}_{class_name2}.png'
            filename = temp2['filename'].values[0]
            print(filename)
            print(new_file_name)
            shutil.copy(filename, f'{savepath}/{model_type}/weirdos/{new_file_name}')
    
            


def feature_vis(savepath, model_type, batch_size):
    train_large = '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/images'
    dataframe = pd.read_csv('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/clus/master/master.csv')
    clases = ['nonTIL_stromal','sTIL','tumor_any','other']
    # map df to classes
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any',3:'other'})
    test_df = dataframe.sample(frac=0.2, random_state=42).iloc[:1]
    outpath = os.path.join(savepath, model_type,"plots", "feature_maps")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # Evaluating the model
    model_dir = os.path.join(savepath, model_type,"model_log")
    model = load_model(f'{model_dir}/model-300-0.70.h5')
    # using only till the global_average_pooling2d layer
    # model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
    model.summary()
    test_data = ImageDataGenerator(rescale = 1.0/255.0)
    test_generator = test_data.flow_from_dataframe(
        dataframe=test_df,
        directory=train_large,
        x_col="image",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(50,50))
    # getting the features
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    feature_map_model = Model(inputs=model.input, outputs=model.get_layer('block1_conv1').output)
    feature_maps = feature_map_model.predict(test_generator)
    # vis
    for ind, feature_map in enumerate(feature_maps):
        print(feature_map.shape)
        plt.figure(figsize=(16,16))
        for i in range(16):
            map = feature_map[:,:,i]
            plt.subplot(4,4,i+1)
            plt.imshow(map)
        plt.savefig(os.path.join(outpath, f'feature_map_{ind}.png'), dpi=300)
        plt.close()

    
     


if __name__ == '__main__':
    # train('InceptionV3', 100, 32, 0.0000001, '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus')
    # train('Xception', 300, 128, 0.0000001, '/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus')
    # gen_stage_wise_conf('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception', 128)
    # plot_model_acc('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception')
    # plot_model_loss('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception')
    # plot_model_conf('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception')
    # test_best('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception', 128)
    # precision_recall('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus/Xception')
    # roc_auc('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus/Xception')
    # feature_gen('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception', 128)
    # gen_few_preds('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception', 128)
    feature_vis('/Users/mraoaakash/Documents/research/aakash-rao-capstone-project/outputs/im_clus', 'Xception', 128)