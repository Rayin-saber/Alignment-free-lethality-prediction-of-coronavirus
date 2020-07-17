import os
import numpy as np
import random
import math
import torch
import torch.nn as nn
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle
from validation import evaluate

#os.chdir('/content/drive/Colab Notebooks/NTU/Corona Virus/code')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reshape_to_linear(x):
    output = np.reshape(x, (x.shape[0], -1))

    return output


# split data into training and testing
def train_test_split_data(feature, label, split_ratio, shuffled_flag):
    setup_seed(20)
    train_x, test_x, train_y, test_y = [], [], [], []
    feature_new, label_new = [], []
    num_of_training = int(math.floor(len(feature) * (1 - split_ratio)))
    if shuffled_flag == True:
        shuffled_index = np.arange(len(feature))
        random.shuffle(shuffled_index)
    else:
        shuffled_index = np.arange(len(feature))
    for i in range(0, len(feature)):
        feature_new.append(feature[shuffled_index[i]])
        label_new.append(label[shuffled_index[i]])

    train_x = feature_new[:num_of_training]
    train_y = label_new[:num_of_training]
    test_x = feature_new[num_of_training:]
    test_y = label_new[num_of_training:]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, test_x, train_y, test_y


def lr_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = linear_model.LogisticRegression().fit(X, Y)
    train_acc = cross_val_score(clf, X, Y, cv=10, scoring='accuracy').mean()
    train_pre = cross_val_score(clf, X, Y, cv=10, scoring='precision_macro').mean()
    train_rec = cross_val_score(clf, X, Y, cv=10, scoring='recall_macro').mean()
    train_fscore = cross_val_score(clf, X, Y, cv=10, scoring='f1_macro').mean()
    # train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    with open('lr.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (train_acc, train_pre, train_rec, train_fscore))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


def knn_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = neighbors.KNeighborsClassifier().fit(X, Y)
    train_acc = cross_val_score(clf, X, Y, cv=10, scoring='accuracy').mean()
    train_pre = cross_val_score(clf, X, Y, cv=10, scoring='precision_macro').mean()
    train_rec = cross_val_score(clf, X, Y, cv=10, scoring='recall_macro').mean()
    train_fscore = cross_val_score(clf, X, Y, cv=10, scoring='f1_macro').mean()
    # train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    with open('knn.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (train_acc, train_pre, train_rec, train_fscore))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


def svm_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = SVC(gamma='auto', class_weight='balanced').fit(X, Y)
    train_acc = cross_val_score(clf, X, Y, cv=10, scoring='accuracy').mean()
    train_pre = cross_val_score(clf, X, Y, cv=10, scoring='precision_macro').mean()
    train_rec = cross_val_score(clf, X, Y, cv=10, scoring='recall_macro').mean()
    train_fscore = cross_val_score(clf, X, Y, cv=10, scoring='f1_macro').mean()
    # train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    with open('svm.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (train_acc, train_pre, train_rec, train_fscore))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


def rf_baseline(X, Y, X_test, Y_test):
    setup_seed(20)
    clf = ensemble.RandomForestClassifier().fit(X, Y)
    train_acc = cross_val_score(clf, X, Y, cv=10, scoring='accuracy').mean()
    train_pre = cross_val_score(clf, X, Y, cv=10, scoring='precision_macro').mean()
    train_rec = cross_val_score(clf, X, Y, cv=10, scoring='recall_macro').mean()
    train_fscore = cross_val_score(clf, X, Y, cv=10, scoring='f1_macro').mean()
    # train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    with open('rf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (train_acc, train_pre, train_rec, train_fscore))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


def nn_baseline(X, Y, X_test, Y_test):
    setup_seed(20)
    clf = MLPClassifier(random_state=100).fit(X, Y)
    train_acc = cross_val_score(clf, X, Y, cv=10, scoring='accuracy').mean()
    train_pre = cross_val_score(clf, X, Y, cv=10, scoring='precision_macro').mean()
    train_rec = cross_val_score(clf, X, Y, cv=10, scoring='recall_macro').mean()
    train_fscore = cross_val_score(clf, X, Y, cv=10, scoring='f1_macro').mean()
    # train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    with open('nn.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f'
          % (train_acc, train_pre, train_rec, train_fscore))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f'
          % (val_acc, precision, recall, fscore))


class CNN_HA(nn.Module):
    def __init__(self):
        super(CNN_HA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 147 * 26, 120)  # overlapped feature dimension
        # self.fc1 = nn.Linear(32 * 50 * 26,120)  #non-overlapped feature dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # 2/3 classes
        # self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # out = self.dropout(out)
        return self.logsoftmax(out)

class AlexNet(nn.Module):
    def __init__(self, model):
        super(AlexNet, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(2304, 256)
        self.Linear_layer2 = nn.Linear(256, 3)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out

class VGG(nn.Module):
    def __init__(self, model):
        super(VGG, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer = nn.Linear(3136, 3)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        out = self.Linear_layer(x)
        out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(512, 256)
        self.Linear_layer2 = nn.Linear(256, 3)
        # self.Linear_layer3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out


class ResNext(nn.Module):
    def __init__(self, model):
        super(ResNext, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(2048, 256)
        self.Linear_layer2 = nn.Linear(256, 3)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out


class Inception(nn.Module):
    def __init__(self, model):
        super(Inception, self).__init__()
        # model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 3)
        self.conv_layer = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.inception_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(739328, 256)
        self.Linear_layer2 = nn.Linear(256, 3)
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.inception_layer(x)
        # x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out


class RnnModel(nn.Module):
    """
    An RNN model using either RNN, LSTM or GRU cells.
    """

    def __init__(self, input_dim, output_dim, hidden_size, hidden_state, dropout_p, cell_type):
        super(RnnModel, self).__init__()

        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        self.dropout = nn.Dropout(dropout_p)

        if cell_type == 'LSTM':
            self.encoder = nn.LSTM(input_dim, hidden_size)
        elif cell_type == 'GRU':
            self.encoder = nn.GRU(input_dim, hidden_size)
        elif cell_type == 'RNN':
            self.encoder = nn.RNN(input_dim, hidden_size)

        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq, hidden_state):
        input_seq = self.dropout(input_seq)
        encoder_outputs, _ = self.encoder(input_seq, None)
        score_seq = self.out(encoder_outputs[-1, :, :])

        dummy_attn_weights = torch.zeros(input_seq.shape[1], input_seq.shape[0])
        return score_seq, dummy_attn_weights  # No attention weights

    def init_hidden(self, batch_size):
        if self.cell_type == 'LSTM':
            h_init = torch.zeros(1, batch_size, self.hidden_size)
            c_init = torch.zeros(1, batch_size, self.hidden_size)
            return (h_init, c_init)
        elif self.cell_type == 'GRU':
            return torch.zeros(1, batch_size, self.hidden_size)
        elif self.cell_type == 'RNN':
            return torch.zeros(1, batch_size, self.hidden_size)