import math
import numpy as np


def get_confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix from given labels and predictions.
    Expects tensors or numpy arrays of same shape.
    """

    ## 3 classes
    TP1, TP2, TP3, FP1, FP2, FP3, TN1, TN2, TN3, FN1, FN2, FN3 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN1 += 1
        elif y_true[i] == 0 and y_pred[i] != 0:
            FP1 += 1
        elif y_true[i] != 0 and y_pred[i] == 0:
            FN1 += 1
        elif y_true[i] != 0 and y_pred[i] != 0:
            TP1 += 1

    for i in range(y_true.shape[0]):
        if y_true[i] == 1 and y_pred[i] == 1:
            TN2 += 1
        elif y_true[i] == 1 and y_pred[i] != 1:
            FP2 += 1
        elif y_true[i] != 1 and y_pred[i] == 1:
            FN2 += 1
        elif y_true[i] != 1 and y_pred[i] != 1:
            TP2 += 1

    for i in range(y_true.shape[0]):
        if y_true[i] == 2 and y_pred[i] == 2:
            TN3 += 1
        elif y_true[i] == 2 and y_pred[i] != 2:
            FP3 += 1
        elif y_true[i] != 2 and y_pred[i] == 2:
            FN3 += 1
        elif y_true[i] != 2 and y_pred[i] != 2:
            TP3 += 1

    conf_matrix1 = [
        [TP1, FP1],
        [FN1, TN1]
    ]
    conf_matrix2 = [
        [TP2, FP2],
        [FN2, TN2]
    ]
    conf_matrix3 = [
        [TP3, FP3],
        [FN3, TN3]
    ]

    return conf_matrix1, conf_matrix2, conf_matrix3


def get_accuracy(conf_matrix1, conf_matrix2, conf_matrix3):
    """
    Calculates macro accuracy metric from the given confusion matrix.
    """
    TP1, FP1, FN1, TN1 = conf_matrix1[0][0], conf_matrix1[0][1], conf_matrix1[1][0], conf_matrix1[1][1]

    TP2, FP2, FN2, TN2 = conf_matrix2[0][0], conf_matrix2[0][1], conf_matrix2[1][0], conf_matrix2[1][1]

    TP3, FP3, FN3, TN3 = conf_matrix3[0][0], conf_matrix3[0][1], conf_matrix3[1][0], conf_matrix3[1][1]

    return (TN1 + TN2 + TN3) / (TP1 + TN1 + FP1 + FN1)


def get_precision(conf_matrix1, conf_matrix2, conf_matrix3):
    """
    Calculates macro precision metric from the given confusion matrix.
    """
    TP1, FP1 = conf_matrix1[0][0], conf_matrix1[0][1]
    TP2, FP2 = conf_matrix2[0][0], conf_matrix2[0][1]
    TP3, FP3 = conf_matrix3[0][0], conf_matrix3[0][1]

    if TP1 + FP1 > 0 and TP2 + FP2 > 0 and TP3 + FP3 > 0:
        return (TP1 / (TP1 + FP1) + TP2 / (TP2 + FP2) + TP3 / (TP3 + FP3)) / 3
    else:
        return 0


def get_recall(conf_matrix1, conf_matrix2, conf_matrix3):
    """
    Calculates macro recall metric from the given confusion matrix.
    """
    TP1, FN1 = conf_matrix1[0][0], conf_matrix1[1][0]
    TP2, FN2 = conf_matrix2[0][0], conf_matrix2[1][0]
    TP3, FN3 = conf_matrix3[0][0], conf_matrix3[1][0]

    if TP1 + FN1 > 0 and TP2 + FN2 > 0 and TP3 + FN3 > 0:
        return (TP1 / (TP1 + FN1) + TP2 / (TP2 + FN2) + TP3 / (TP3 + FN3)) / 3
    else:
        return 0


def get_f1score(conf_matrix1, conf_matrix2, conf_matrix3):
    """
    Calculates macro f1-score metric from the given confusion matrix.
    """
    p = get_precision(conf_matrix1, conf_matrix2, conf_matrix3)
    r = get_recall(conf_matrix1, conf_matrix2, conf_matrix3)

    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def get_mcc(conf_matrix1, conf_matrix2, conf_matrix3):
    """
    Calculates macro Matthew's Correlation Coefficient metric from the given confusion matrix.
    """
    TP1, FP1, FN1, TN1 = conf_matrix1[0][0], conf_matrix1[0][1], conf_matrix1[1][0], conf_matrix1[1][1]
    TP2, FP2, FN2, TN2 = conf_matrix2[0][0], conf_matrix2[0][1], conf_matrix2[1][0], conf_matrix2[1][1]
    TP3, FP3, FN3, TN3 = conf_matrix1[0][0], conf_matrix3[0][1], conf_matrix3[1][0], conf_matrix3[1][1]

    if TP1 + FP1 > 0 and TP1 + FN1 > 0 and TN1 + FP1 > 0 and TN1 + FN1 > 0 and TP2 + FP2 > 0 and TP2 + FN2 > 0 and TN2 + FP2 > 0 and TN2 + FN2 > 0 and TP3 + FP3 > 0 and TP3 + FN3 > 0 and TN3 + FP3 > 0 and TN3 + FN3 > 0:
        return ((TP1 * TN1 - FP1 * FN1) / math.sqrt((TP1 + FP1) * (TP1 + FN1) * (TN1 + FP1) * (TN1 + FN1)) + (
                    TP2 * TN2 - FP2 * FN2) / math.sqrt((TP2 + FP2) * (TP2 + FN2) * (TN2 + FP2) * (TN2 + FN2)) + (
                            TP3 * TN3 - FP3 * FN3) / math.sqrt(
            (TP3 + FP3) * (TP3 + FN3) * (TN3 + FP3) * (TN3 + FN3))) / 3
    else:
        return 0


def evaluate(Y_real, Y_pred):
    conf_matrix1, conf_matrix2, conf_matrix3 = get_confusion_matrix(Y_real, Y_pred)
    precision = get_precision(conf_matrix1, conf_matrix2, conf_matrix3)
    recall = get_recall(conf_matrix1, conf_matrix2, conf_matrix3)
    fscore = get_f1score(conf_matrix1, conf_matrix2, conf_matrix3)
    mcc = get_mcc(conf_matrix1, conf_matrix2, conf_matrix3)
    val_acc = get_accuracy(conf_matrix1, conf_matrix2, conf_matrix3)

    return precision, recall, fscore, mcc, val_acc


def list_summary(name, data):
    print(name)
    unique, count = np.unique(data, return_counts=True)
    print(dict(zip(unique, count)))


def get_time_string(time):
    """
    Creates a string representation of minutes and seconds from the given time.
    """
    mins = time // 60
    secs = time % 60
    time_string = ''

    if mins < 10:
        time_string += '  '
    elif mins < 100:
        time_string += ' '

    time_string += '%dm ' % mins

    if secs < 10:
        time_string += ' '

    time_string += '%ds' % secs

    return time_string