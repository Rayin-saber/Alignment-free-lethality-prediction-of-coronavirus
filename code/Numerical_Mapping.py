import os
import pandas as pd
from scipy.fftpack import fft
from CGR import *
import numpy as np
import length_Calc

def numerical_mapping(PATH, k):
    path = PATH
    files = os.listdir(path)
    temp_feature_train, temp_feature_test = [], []
    #medLen = int(length_Calc.length_calc(path))  # get the normalized length
    medLen = 172*172  # 468*64
    print("the median length is:", medLen)
    label_train, label_test = [], []
    if k == 'Integer':
        mapping = IntegerMapping
    elif k == 'NN':
        mapping = NNMapping
    elif k =='EIIP':
        mapping = EIIPMapping
    elif k =='PP':
        mapping = PPMapping
    elif k =='Real':
        mapping = RealMapping
    else:
        mapping = Just_AMapping

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
                    if len(sequence) < 20000:
                        pass
                    else:
                        DNASeq = seq_replace(sequence)
                        num_seq = mapping(DNASeq)  # numerical representation
                        if len(num_seq) > medLen:
                            new_seq = num_seq[:medLen]
                        elif len(num_seq) < medLen:
                            new_seq = num_seq + [0] * (medLen - len(num_seq))
                        else:
                            new_seq = num_seq
                        #print("flag")
                        DFT_seq = [round(i, 2) for i in abs(fft(new_seq))]  # fft operation
                        if file.split("_")[-1] == "train.fasta":
                            temp_feature_train.append(DFT_seq)
                            #print(key, " is done!")
                            if file.split("_")[0] == "COVID19":
                                label_train.append(0)
                            elif file.split("_")[0] == "SARS":
                                label_train.append(1)
                            else:
                                label_train.append(2)
                        elif file.split("_")[-1] == "test.fasta":
                            temp_feature_test.append(DFT_seq)
                            #print(key, " is done!")
                            if file.split("_")[0] == "COVID19":
                                label_test.append(0)
                            elif file.split("_")[0] == "SARS":
                                label_test.append(1)
                            else:
                                label_test.append(2)
        print(file, " is done!")


    # feature = pd.DataFrame(data=temp_feature)
    # temp_label = pd.DataFrame(data=label)
    # feature.to_csv(path + "/" + "feature_IntegerMapping_after.csv", encoding='utf-8', index=False, header=False)
    # #temp_label.to_csv(path + "/" + "label_after.csv", encoding='utf-8')
    return np.array(temp_feature_train), np.array(label_train), np.array(temp_feature_test), np.array(label_test)

