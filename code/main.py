from model import *
from train_CNN import train_cnn
import torchvision.models as models
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import torch
import iCGR
from Numerical_Mapping import numerical_mapping
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def main(feature_train, label_train, feature_test, label_test, k):
    SHUFFLE_FLAG = True  # Whether random the data set

    METHODS = ['Traditional', 'CNN', 'RNN']
    METHOD = METHODS[0]

    CNNs = ['VGG19', 'RESNET34', 'AlexNet']
    RNNs = ['LSTM', 'GRU']
    NET = CNNs[2]

    Classifiers = ['KNN', 'LR', 'NN', 'RF']
    MODEL = Classifiers[0]

    scaler = preprocessing.StandardScaler()
    feature_train = scaler.fit_transform(feature_train)
    feature_test = scaler.fit_transform(feature_test)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)

    for ID in range(4):
        MODEL = Classifiers[ID]
        if METHOD == 'CNN':
            parameters = {
                # Note, no learning rate decay implemented
                'learning_rate': 0.0001,

                # Size of mini batch
                'batch_size': 64,

                # Number of training iterations
                'num_of_epochs': 50
            }

            feature_train_new, label_train_new, feature_test_new, label_test_new = [], [], [], []
            if SHUFFLE_FLAG == True:
                shuffled_index = np.arange(len(feature_train))
                random.shuffle(shuffled_index)
            else:
                shuffled_index = np.arange(len(feature_train))
            for i in range(0, len(feature_train)):
                feature_train_new.append(feature_train[shuffled_index[i]])
                label_train_new.append(label_train[shuffled_index[i]])
            if SHUFFLE_FLAG == True:
                shuffled_index = np.arange(len(feature_test))
                random.shuffle(shuffled_index)
            else:
                shuffled_index = np.arange(len(feature_test))
            for i in range(0, len(feature_test)):
                feature_test_new.append(feature_test[shuffled_index[i]])
                label_test_new.append(label_test[shuffled_index[i]])

            if feature_transform == 'DFT':
                train_x = np.reshape(feature_train_new, (np.array(feature_train_new).shape[0], 1, 172, 172))
                test_x = np.reshape(feature_test_new, (np.array(feature_test_new).shape[0], 1, 172, 172))
            else:
                train_x = np.reshape(feature_train_new, (np.array(feature_train_new).shape[0], 1, pow(2, k), pow(2, k)))
                test_x = np.reshape(feature_test_new, (np.array(feature_test_new).shape[0], 1, pow(2, k), pow(2, k)))
            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(np.array(label_train_new), dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(np.array(label_test_new), dtype=torch.int64).cuda()
            # print(train_y[:20], test_y[:20])
            if NET == 'RESNET34':
                net = ResNet(models.resnet34(pretrained=True))
                print("Using ResNet34...")
            elif NET == 'VGG19':
                net = VGG(models.vgg19_bn(pretrained=True))
                print("Using VGG19...")
            elif NET == 'AlexNet':
                net = AlexNet(models.alexnet(pretrained=True))
                print("Using AlexNet...")
            if torch.cuda.is_available():
                print('running with GPU')
                net.cuda()

            train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                      train_y, test_x,
                      test_y, False)

        elif METHOD == 'Traditional':
            feature_train_new, label_train_new, feature_test_new, label_test_new = [], [], [], []
            if SHUFFLE_FLAG == True:
                shuffled_index = np.arange(len(feature_train))
                random.shuffle(shuffled_index)
            else:
                shuffled_index = np.arange(len(feature_train))
            for i in range(0, len(feature_train)):
                feature_train_new.append(feature_train[shuffled_index[i]])
                label_train_new.append(label_train[shuffled_index[i]])
            if SHUFFLE_FLAG == True:
                shuffled_index = np.arange(len(feature_test))
                random.shuffle(shuffled_index)
            else:
                shuffled_index = np.arange(len(feature_test))
            for i in range(0, len(feature_test)):
                feature_test_new.append(feature_test[shuffled_index[i]])
                label_test_new.append(label_test[shuffled_index[i]])
            if MODEL == 'LR':
                print('Using LR...')
                lr_baseline(reshape_to_linear(np.array(feature_train_new)), np.array(label_train_new),
                            reshape_to_linear(np.array(feature_test_new)), np.array(label_test_new))
            elif MODEL == 'RF':
                print('Using RF...')
                rf_baseline(reshape_to_linear(np.array(feature_train_new)), np.array(label_train_new),
                            reshape_to_linear(np.array(feature_test_new)), np.array(label_test_new))
            elif MODEL == 'KNN':
                print('Using KNN...')
                knn_baseline(reshape_to_linear(np.array(feature_train_new)), np.array(label_train_new),
                             reshape_to_linear(np.array(feature_test_new)), np.array(label_test_new))
            elif MODEL == 'NN':
                print('Using NN...')
                nn_baseline(reshape_to_linear(np.array(feature_train_new)), np.array(label_train_new),
                            reshape_to_linear(np.array(feature_test_new)), np.array(label_test_new))
            elif MODEL == 'SVM':
                print('Using SVM...')
                svm_baseline(reshape_to_linear(np.array(feature_train_new)), np.array(label_train_new),
                             reshape_to_linear(np.array(feature_test_new)), np.array(label_test_new))


PATH = "E:/2019 NTU/Project/Program/Corona Virus/data/nucleotide/time series"
#os.chdir("/content/drive/Colab Notebooks/NTU/Corona Virus/data/nucleotide/Time_series/")
while True:
    print("Please input the feature transform method:")
    feature_transform = input()
    if feature_transform == 'DFT':
        print("Using DFT...")
        k = ["Real", "Integer", "NN", "EIIP", "PP", "Just_A"]
        # k = ["Just_A"]
        for item in k:
            print("****************************************************************************")
            print("Using", item)
            feature_train, label_train, feature_test, label_test = numerical_mapping(PATH, item)  # Using DFT
            main(feature_train, label_train, feature_test, label_test, item)
        break
    elif feature_transform == 'CGR':
        print("Using CGR...")
        k = range(1, 7)
        for item in k:
            print("****************************************************************************")
            print("Using", str(item), "mers")
            feature_train, label_train, feature_test, label_test = iCGR.iCGR(PATH, item)  # Using CGR
            main(feature_train, label_train, feature_test, label_test, item)
        break
    else:
        print("Input Error, please try again!")
        continue

