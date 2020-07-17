import os, sys
import numpy as np
from tkinter import _flatten
import CGR
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels


k = 9
pca = PCA(n_components=3)
ica = FastICA(n_components=3, random_state=12)
scaler = preprocessing.StandardScaler()


def iCGR_draw(PATH, k):
    #path = "E:/2019 NTU/Project/Program/Corona Virus/data/nucleotide"
    path = PATH
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature_all = [], [], [], [], [], [], [], []

    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
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
                        DNASeq = CGR.seq_replace(sequence)
                        x, y, n = CGR.encodeDNASequence(DNASeq, k)
                        figure = [[0 for row in range(pow(2, k))] for col in range(pow(2, k))]
                        # print(feature)
                        for x_loc, y_loc in zip(x, y):
                            # print(x_loc)
                            figure[x_loc + pow(2, k-1) - 1][y_loc + pow(2, k-1) - 1] += 1
                        feature_all.append(list(_flatten(figure)))

                        if file.split(".")[0] == "COVID19":
                            feature1.append(list(_flatten(figure)))
                        elif file.split(".")[0] == "SARS":
                            feature2.append(list(_flatten(figure)))
                        elif file.split(".")[0] == "MERS":
                            feature3.append(list(_flatten(figure)))
                        elif file.split(".")[0] == "HCoV-229E":
                            feature4.append(list(_flatten(figure)))
                        elif file.split(".")[0] == "HCoV-HKU1":
                            feature5.append(list(_flatten(figure)))
                        elif file.split(".")[0] == "HCoV-NL63":
                            feature6.append(list(_flatten(figure)))
                        else:
                            feature7.append(list(_flatten(figure)))
    # temp_feature = pd.DataFrame(data=feature)
    # temp_label = pd.DataFrame(data=label)
    # temp_feature.to_csv(path + "/" + "feature_" + str(k) +"mers_after.csv", encoding='utf-8', index=False, header=False)
    # #temp_label.to_csv(path + "/" + "label_after.csv", encoding='utf-8')
    return np.array(feature1), np.array(feature2), np.array(feature3), np.array(feature4), np.array(feature5), np.array(feature6), np.array(feature7), np.array(feature_all)

PATH = "E:/2019 NTU/Project/Program/Corona Virus/data/nucleotide/subtype"

distances = ['cityblock', 'manhattan', 'euclidean', 'l1', 'l2', 'cosine']
kernels = ['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']
distance = distances[1]
kernel = kernels[5]

feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature_all = iCGR_draw(PATH, k)  # Using CGR


feature1 = pairwise_distances(feature1, metric=distance)
feature1 = ica.fit_transform(feature1)
feature1 = scaler.fit_transform(feature1)
feature2 = pairwise_distances(feature2, metric=distance)
feature2 = ica.fit_transform(feature2)
feature2 = scaler.fit_transform(feature2)
feature3 = pairwise_distances(feature3, metric=distance)
feature3 = ica.fit_transform(feature3)
feature3 = scaler.fit_transform(feature3)
feature4 = pairwise_distances(feature4, metric=distance)
feature4 = ica.fit_transform(feature4)
feature4 = scaler.fit_transform(feature4)
feature5 = pairwise_distances(feature5, metric=distance)
feature5 = ica.fit_transform(feature5)
feature5 = scaler.fit_transform(feature5)
feature6 = pairwise_distances(feature6, metric=distance)
feature6 = ica.fit_transform(feature6)
feature6 = scaler.fit_transform(feature6)
feature7 = pairwise_distances(feature7, metric=distance)
feature7 = ica.fit_transform(feature7)
feature7 = scaler.fit_transform(feature7)
# print(feature3)
fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(feature1[:, 0], feature1[:, 1], feature1[:, 2], marker='o', color = 'pink', label='SARS-CoV2')
ax.scatter(feature2[:, 0], feature2[:, 1], feature2[:, 2], marker='o', color = 'red', label='SARS-CoV')
ax.scatter(feature3[:, 0], feature3[:, 1], feature3[:, 2], marker='o', color = 'blue', label='MERS-CoV')
ax.scatter(feature4[:, 0], feature4[:, 1], feature4[:, 2], marker='o', color = 'yellow', label='HCoV-229E')
ax.scatter(feature5[:, 0], feature5[:, 1], feature5[:, 2], marker='o', color = 'orange', label='HCoV-HKU1')
ax.scatter(feature6[:, 0], feature6[:, 1], feature6[:, 2], marker='o', color = 'green', label='HCoV-NL63')
ax.scatter(feature7[:, 0], feature7[:, 1], feature7[:, 2], marker='o', color = 'purple', label='HCoV-OC43')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.set_xlim(-5, 10)
# ax.set_xlim(-5, 15)
# ax.set_xlim(-5, 15)
plt.legend(loc='best')
plt.show()