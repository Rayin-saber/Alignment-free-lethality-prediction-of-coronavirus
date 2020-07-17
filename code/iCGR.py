import CGR
import os
import pandas as pd
import numpy as np
from tkinter import _flatten



def iCGR(PATH, k):
    #path = "E:/2019 NTU/Project/Program/Corona Virus/data/nucleotide"
    path = PATH
    files = os.listdir(path)
    feature_train = []
    label_train = []
    feature_test = []
    feature = []
    label_test = []

    for file in files:
        if not os.path.isdir(file):
            if file.split(".")[-1] == "fasta":
                fasta = open(path + "/" + file)
                seq = {}
                for line in fasta:
                    if line.startswith('>'):
                        name = line.replace('>', '').split()[0]
                        seq[name] = ''
                    else:
                        seq[name] += line.replace('\n', '').strip()
                for key in seq:
                    sequence = seq[key]
                    if len(sequence) < 20000:  # when the length is too small, pass
                        pass
                    else:
                        DNASeq = CGR.seq_replace(sequence)
                        x, y, n = CGR.encodeDNASequence(DNASeq, k)  # encoding the sequence through CGR
                        figure = [[0 for row in range(pow(2, k))] for col in range(pow(2, k))]
                        # print(feature)
                        for x_loc, y_loc in zip(x, y):
                            # print(x_loc)
                            figure[x_loc + pow(2, k-1) - 1][y_loc + pow(2, k-1) - 1] += 1
                        if file.split("_")[-1] == "train.fasta":
                            feature_train.append(list(_flatten(figure)))
                            if file.split("_")[0] == "COVID19":
                                label_train.append(0)
                            elif file.split("_")[0] == "SARS":
                                label_train.append(1)
                            else:
                                label_train.append(2)
                        elif file.split("_")[-1] == "test.fasta":
                            feature_test.append(list(_flatten(figure)))
                            if file.split("_")[0] == "COVID19":
                                label_test.append(0)
                            elif file.split("_")[0] == "SARS":
                                label_test.append(1)
                            else:
                                label_test.append(2)
                        else:  # for use of loading model only
                            feature.append(list(_flatten(figure)))
    # temp_feature = pd.DataFrame(data=feature)
    # temp_label = pd.DataFrame(data=label)
    # temp_feature.to_csv(path + "/" + "feature_" + str(k) +"mers_after.csv", encoding='utf-8', index=False, header=False)
    # #temp_label.to_csv(path + "/" + "label_after.csv", encoding='utf-8')
    return np.array(feature_train), np.array(label_train), np.array(feature_test), np.array(label_test)
    #return np.array(feature)


