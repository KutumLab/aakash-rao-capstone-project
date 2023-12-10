import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
def train(train_large,path, model_type, epochs, batch_size,learning_rate, suffix=""):
    path = os.path.join(path,'models')
    dataframe = pd.read_csv(os.path.join(train_large, 'master.csv'))
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any'})
    train_large = os.path.join(train_large, "images")
    #checking tensorflow version
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #(trainX, testX, trainY, testY) = train_test_split(data, train_large.target, test_size=0.25)
    savepath = os.path.join(path, model_type+suffix)

    print(f'Model Type: {model_type}')
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
    print(len(class_keys))
    class_keys = valid_generator.class_indices.keys()
    print(class_keys)


    if model_type == 'InceptionV3':
        inception = InceptionV3( 
            weights='imagenet', 
            include_top=False, 
            input_shape=(300,300,3) 
            )
        for layer in inception.layers:
            layer.trainable = True
        x = layers.Flatten()(inception.output)
        # adding avgpool layer
        x = layers.GlobalAveragePooling2D()(inception.output)
        x = layers.Dense(1024, activation = 'relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dense(3, activation = 'softmax', kernel_regularizer=l2(0.01))(x)
        model = Model(inception.input, x)
        model.compile(optimizer = RMSprop(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    # Creating the model
    if model_type == 'InceptionV3':
            inception = InceptionV3(
                    weights=None,
                    include_top=False,
                    input_shape=(300,300,3)
                    )
            for layer in inception.layers:
                    layer.trainable = True

            x = layers.GlobalAveragePooling2D()(inception.output)
            x = layers.Dense(128, activation = 'relu', kernel_regularizer=l2(0.01))(x)
            x = layers.Dense(len(class_keys), activation = 'softmax',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
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

            x = layers.GlobalAveragePooling2D()(xception.output)
            x = layers.Dense(128, activation = 'relu', kernel_regularizer=l2(0.01))(x)
            x = layers.Dense(len(class_keys), activation = 'softmax',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
            model = Model(xception.input, x)
            model.compile(optimizer = RMSprop(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    if not os.path.exists(f'{savepath}'):
        os.makedirs(f'{savepath}')


    #TF_CPP_MIN_LOG_LEVEL=2
    # Training the model

    print("------------------------------------------")
    print(f'Training the model {model_type}')
    print("------------------------------------------")
    filepath = f'{savepath}/model_log'
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
    model.save(f'{savepath}/{model_type+suffix}.h5')
    print("------------------------------------------")
    print(f'Model saved')
    print("------------------------------------------")


    #plotting the accuracy and loss
    print("------------------------------------------")
    print(f'Plotting and supplimentary data')
    print("------------------------------------------")

    hist_df = pd.DataFrame(history.history) 
    hist_json_file = f'{savepath}/history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv: 
    hist_csv_file = f'{savepath}/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        
    loaded_model = load_model(f'{savepath}/{model_type}.h5')
    outcomes = loaded_model.predict(valid_generator)
    y_pred = np.argmax(outcomes, axis=1)
    # confusion matrix
    confusion = confusion_matrix(valid_generator.classes, y_pred)


    conf_df = pd.DataFrame(confusion, index = class_keys, columns = class_keys)
    conf_df.to_csv(f'{savepath}/Confusion_matrix.csv')

    # classification report
    target_names = class_keys
    report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'{savepath}/Classification_report.csv')

    print("------------------------------------------")
    print(f'Supplimentary Data Saved')
    print("------------------------------------------")


    print("------------------------------------------")
    print(f'Model Evaluation')
    print("------------------------------------------")

    # Evaluating the model
    model = load_model(f'{savepath}/{model_type}.h5')
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
    pred_df.to_csv(f'{savepath}/Predictions.csv')

    y_truth = test_generator.classes
    y_pred = np.argmax(pred, axis=1)
    confusion = confusion_matrix(y_truth, y_pred)

    conf_df = pd.DataFrame(confusion, index = class_keys, columns = class_keys)
    conf_df.to_csv(f'{savepath}/Confusion_matrix_test.csv')


def test_best(train_large,path, model_type, epochs, batch_size,learning_rate, suffix=""):
    dataframe = pd.read_csv(os.path.join(train_large,'master.csv'))
    train_large = os.path.join(train_large, "images")
    # map df to classes


    model_path = os.path.join(path, "models", model_type+suffix)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)
    npy_savepath = os.path.join(path, "npys", model_type+suffix)
    os.makedirs(npy_savepath, exist_ok=True)
    plot_path = os.path.join(path, "plots", model_type+suffix)

    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any',})
    test_df = dataframe.sample(frac=0.2, random_state=42)
    # Evaluating the model
    model = load_model(os.path.join(model_path, model_type+suffix+".h5"))
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

    gt_labels = test_generator.classes
    pred = model.predict(test_generator)
    pred_df = pd.DataFrame(pred)
    test_pred = pred_df 
    pred_cls = []
    for indes, row in test_pred.iterrows():
        pred_cls.append(row.values.argmax())
    test_pred['gt'] =gt_labels
    test_pred['pred'] = pred_cls

    test_pred.columns = ['nonTIL_stromal','sTIL','tumor_any', 'gt','pred']
    print(test_pred.head())
    print(len(test_pred[test_pred['gt']==test_pred['pred']])/test_pred.shape[0])
    test_pred.to_csv(f'{csv_savepath}/gt_pred.csv')



def gen_stage_wise_conf(path, model_type, train_large, suffix="", batch_size=32):
    model_path = os.path.join(path, "models", model_type+suffix)
    plot_savepath = os.path.join(path, "plots", model_type+suffix)
    os.makedirs(plot_savepath, exist_ok=True)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)
    npy_savepath = os.path.join(path, "npys", model_type+suffix)
    os.makedirs(npy_savepath, exist_ok=True)
    dataframe = pd.read_csv(os.path.join(train_large, 'master.csv'))
    train_large = os.path.join(train_large, "images")
    # map df to classes
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any'})
    test_df = dataframe.sample(frac=0.2, random_state=42)
    test_df.to_csv(os.path.join(csv_savepath, 'test.csv'))
    outpath = os.path.join(csv_savepath, "epoch_wise_conf")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # Evaluating the model
    model_dir = os.path.join(model_path,"model_log")
    overall_acc = []
    overal_loss = []
    model_epochs = []
    for epoch_model in os.listdir(model_dir):
        if '.DS_Store' in epoch_model:
            continue
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
    df.to_csv(os.path.join(csv_savepath, 'epoch_wise_acc.csv'))


def plot_model_acc(path, model_type,suffix=""):
    
    line_width = 0.85
    font_size = 14

    model_path = os.path.join(path, "models", model_type+suffix)
    plot_path = os.path.join(path, "plots", model_type+suffix)
    os.makedirs(plot_path, exist_ok=True)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)
    npy_savepath = os.path.join(path, "npys", model_type+suffix)

    history = pd.read_csv(os.path.join(model_path, 'history.csv'))
    history.columns = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
    epoch_wise_acc = pd.read_csv(os.path.join(csv_savepath, 'epoch_wise_acc.csv'))

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


def plot_model_loss(path, model_type,suffix=""):
    
    line_width = 0.85
    font_size = 14

    model_path = os.path.join(path, "models", model_type+suffix)
    plot_path = os.path.join(path, "plots", model_type+suffix)
    os.makedirs(plot_path, exist_ok=True)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)
    npy_savepath = os.path.join(path, "npys", model_type+suffix)

    history = pd.read_csv(os.path.join(model_path, 'history.csv'))
    history.columns = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
    epoch_wise_acc = pd.read_csv(os.path.join(csv_savepath, 'epoch_wise_acc.csv'))

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

def plot_model_conf(path, model_type,suffix=""):
    line_width = 0.85
    font_size = 14

    model_path = os.path.join(path, "models", model_type+suffix)
    plot_path = os.path.join(path, "plots", model_type+suffix)
    os.makedirs(plot_path, exist_ok=True)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)
    npy_savepath = os.path.join(path, "npys", model_type+suffix)

    matrix = pd.read_csv(os.path.join(model_path, 'Confusion_matrix_test.csv'))
    classes = ['Stromal','sTIL','Tumor']
    matrix = matrix.drop(columns=['Unnamed: 0'])
    matrix.columns = classes
    matrix.index = classes
    print(matrix.head())
    for col in classes:
        matrix[col] = round(matrix[col]/matrix[col].sum()*100,2)
    print(matrix.head())
    plt.figure(figsize=(4,4))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=font_size, fontweight='bold')
    plt.xlabel('Ground Truth Labels', fontsize=font_size, fontweight='bold')
    plt.ylabel('Predicted Labels', fontsize=font_size, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # custom legend
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(plot_path,'confusion.png'), dpi=300)
    plt.close()
    pass


def roc_auc(path, model_type,suffix=""):
    line_width = 0.85
    font_size = 14

    model_path = os.path.join(path, "models", model_type+suffix)
    plot_path = os.path.join(path, "plots", model_type+suffix)
    os.makedirs(plot_path, exist_ok=True)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)
    npy_savepath = os.path.join(path, "npys", model_type+suffix)


    test_pred = pd.read_csv(os.path.join(csv_savepath, 'gt_pred.csv'))
    test_pred = test_pred.drop(columns=['Unnamed: 0'])
    test_pred.columns = [0,1,2,'gt','pred']
    print(test_pred.head())

    # generating precision recall curve
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y = label_binarize(test_pred['gt'], classes=[0, 1, 2])
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
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.legend(loc='lower right', fontsize=10)
    plt.xlabel('False Positive Rate', fontsize=font_size, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=font_size, fontweight='bold')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve', fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path,'roc_auc.png'), dpi=300)
    # plt.show()
    plt.close()
    pass


def gen_GAP_features(train_large, path, model_type, batch_size, suffix=""):
    line_width = 0.85
    font_size = 14

    model_path = os.path.join(path, "models", model_type+suffix)
    plot_path = os.path.join(path, "plots", model_type+suffix)
    os.makedirs(plot_path, exist_ok=True)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)
    npy_savepath = os.path.join(path, "npys", model_type+suffix)
    os.makedirs(npy_savepath, exist_ok=True)

    dataframe = pd.read_csv(os.path.join(train_large, 'master.csv'))
    train_large = os.path.join(train_large, "images")
    # map df to classes
    dataframe['label'] = dataframe['label'].map({0:'nonTIL_stromal',1:'sTIL',2:'tumor_any'})
    extra_data_generator = ImageDataGenerator(rescale = 1.0/255.0)
    extra_generator = extra_data_generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=train_large,
        x_col="image",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(50,50))
    model = load_model(os.path.join(model_path, model_type+suffix+".h5"))
    model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
    extra_features = model.predict(extra_generator)
    print(extra_features.shape)
    np.save(os.path.join(npy_savepath, 'all_GAP_features.npy'), extra_features)

def GAP_feature_TSNE(train_large, path, model_type, batch_size, suffix=""):
    line_width = 0.85
    font_size = 14

    npy_savepath = os.path.join(path, "npys", model_type+suffix)
    if not os.path.exists(os.path.join(npy_savepath, 'all_GAP_features.npy')):
        gen_GAP_features(train_large, path, model_type, batch_size, suffix)
    X = np.load(os.path.join(npy_savepath, 'all_GAP_features.npy'))

    plot_savepath = os.path.join(path, "plots", model_type+suffix)
    os.makedirs(plot_savepath, exist_ok=True)
    csv_savepath = os.path.join(path, "csvs", model_type+suffix)
    os.makedirs(csv_savepath, exist_ok=True)

    labels = pd.read_csv(os.path.join(train_large, 'master.csv'))
    labels = labels['label'].values
    print(X.shape)

    class_1_indices = np.where(np.array(labels)==0)[0]
    class_2_indices = np.where(np.array(labels)==1)[0]
    class_3_indices = np.where(np.array(labels)==2)[0]


    if not os.path.exists(os.path.join(csv_savepath,'GAP_Feature_tsne.csv')):
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(X)

        tsne_df = pd.DataFrame({'TSNE-1':tsne_results[:,0], 'TSNE-2':tsne_results[:,1], 'class':labels})
        tsne_df.to_csv(os.path.join(csv_savepath,'GAP_Feature_tsne.csv'), index=False)
    else:
        print('Loading t-sne results from csv')
        tsne_df = pd.read_csv(os.path.join(csv_savepath,'GAP_Feature_tsne.csv'))

    # plotting
    plt.figure(figsize=(4,4))
    cls_1 = plt.scatter(tsne_df.iloc[class_1_indices,0], tsne_df.iloc[class_1_indices,1], label='Stromal',marker='o', s=0.25)
    cls_2 = plt.scatter(tsne_df.iloc[class_2_indices,0], tsne_df.iloc[class_2_indices,1], label='sTIL',marker='o', s=0.25)
    cls_3 = plt.scatter(tsne_df.iloc[class_3_indices,0], tsne_df.iloc[class_3_indices,1], label='Tumor',marker='o', s=0.25)

    plt.legend(handles=[cls_1, cls_2, cls_3], loc='upper right', fontsize=6)
    plt.title('Xception Features from\nPooling Layer (T-SNE)', fontsize=font_size, fontweight='bold')
    plt.xlabel('1st component', fontsize=font_size, fontweight='bold')
    plt.ylabel('2nd component', fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_savepath,'GAP_featur_cluster.png'), dpi=300)
    plt.close()

     


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train')
    argparser.add_argument('-t', '--train_large', type=str, help='Path to the input data')
    argparser.add_argument('-p', '--path', type=str, help='Path to save the extracted data')
    argparser.add_argument('-m', '--model_type', type=str, help='Model Type')
    argparser.add_argument('-e', '--epochs', type=int, help='Epochs')
    argparser.add_argument('-b', '--batch_size', type=int, help='Batch Size')
    argparser.add_argument('-l', '--learning_rate', type=float, help='Learning Rate')
    args = argparser.parse_args()
    # train(args.train_large, args.path, args.model_type, args.epochs, args.batch_size, args.learning_rate, suffix="_three_class")
    test_best(args.train_large, args.path, args.model_type, args.epochs, args.batch_size, args.learning_rate, suffix="_three_class")
    gen_stage_wise_conf(args.path, args.model_type, args.train_large, suffix="_three_class", batch_size=args.batch_size)
    plot_model_acc(args.path, args.model_type, suffix="_three_class")
    plot_model_loss(args.path, args.model_type, suffix="_three_class")
    plot_model_conf(args.path, args.model_type, suffix="_three_class")
    roc_auc(args.path, args.model_type, suffix="_three_class")
    gen_GAP_features(args.train_large, args.path, args.model_type, args.batch_size, suffix="_three_class")
    GAP_feature_TSNE(args.train_large, args.path, args.model_type, args.batch_size, suffix="_three_class")