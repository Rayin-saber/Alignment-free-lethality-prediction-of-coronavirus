import os
import matplotlib.pyplot as plt
from CGR import seq_replace
import numpy as np
from matplotlib.ticker import FuncFormatter

def to_percent(temp, position):
    return '%2.0f'%(100*temp) + '%'

k = 7  # k mers
path = "E:/2019 NTU/Project/Program/Corona Virus/data/nucleotide"
files = os.listdir(path)
COVID_19 = []
SARS = []
MERS = []

corona_A, corona_C, corona_T, corona_G = [], [], [], []

for file in files:
    if not os.path.isdir(file):
        if file.split(".")[-1] == "fasta":
            if file.split(".")[0] == "COVID19" or file.split(".")[0] == "SARS" or file.split(".")[0] == "MERS":
                ratio_A, ratio_C, ratio_T, ratio_G = [], [], [], []
                fasta = open(path + "/" + file)
                seq = {}
                for line in fasta:
                    if line.startswith('>'):
                        name = line.replace('>', '').split()[0]
                        seq[name] = ''
                    else:
                        seq[name] += line.replace('\n', '').strip()
                for i, key in enumerate(seq):
                    num_A, num_C, num_T, num_G = 0, 0, 0, 0
                    sequence = seq[key]
                    if len(sequence) < 20000:
                        pass
                    else:
                        DNASeq = seq_replace(sequence)
                        for nucleotide in DNASeq:
                            if nucleotide == 'A':
                                num_A += 1
                            elif nucleotide == 'T':
                                num_T += 1
                            elif nucleotide == 'G':
                                num_G += 1
                            else:
                                num_C += 1
                        # if num_C == 0:
                        #     print(i)
                        ratio_A.append(num_A / len(sequence))
                        ratio_C.append(num_C / len(sequence))
                        ratio_T.append(num_T / len(sequence))
                        ratio_G.append(num_G / len(sequence))

                corona_A.append(ratio_A)
                corona_C.append(ratio_C)
                corona_T.append(ratio_T)
                corona_G.append(ratio_G)
        else:
            pass

bw = 0.2
fig = plt.figure()
ax = plt.subplot()
index = np.arange(3)
dataA = [np.mean(corona_A[0]), np.mean(corona_A[1]), np.mean(corona_A[2])]
print("COVID19_A: %.3f - %.3f" % (min(corona_A[0]),  max(corona_A[0])))
print("COVID19_C: %.3f - %.3f" % (min(corona_C[0]), max(corona_C[0])))
print("COVID19_T: %.3f - %.3f" % (min(corona_T[0]), max(corona_T[0])))
print("COVID19_G: %.3f - %.3f" % (min(corona_G[0]), max(corona_G[0])))
print("MERS_A: %.3f - %.3f" % (min(corona_A[1]), max(corona_A[1])))
print("MERS_C: %.3f - %.3f" % (min(corona_T[1]), max(corona_C[1])))
print("MERS_T: %.3f - %.3f" % (min(corona_C[1]), max(corona_T[1])))
print("MERS_G: %.3f - %.3f" % (min(corona_G[1]), max(corona_G[1])))
print("SARS_A: %.3f - %.3f" % (min(corona_A[2]), max(corona_A[2])))
print("SARS_C: %.3f - %.3f" % (min(corona_C[2]), max(corona_C[2])))
print("SARS_T: %.3f - %.3f" % (min(corona_T[2]), max(corona_T[2])))
print("SARS_G: %.3f - %.3f" % (min(corona_G[2]), max(corona_G[2])))

stdA = [np.std(corona_A[0]), np.std(corona_A[1]), np.std(corona_A[2])]
dataC = [np.mean(corona_C[0]), np.mean(corona_C[1]), np.mean(corona_C[2])]
stdC = [np.std(corona_C[0]), np.std(corona_C[1]), np.std(corona_C[2])]
dataT = [np.mean(corona_T[0]), np.mean(corona_T[1]), np.mean(corona_T[2])]
stdT = [np.std(corona_T[0]), np.std(corona_T[1]), np.std(corona_T[2])]
dataG = [np.mean(corona_G[0]), np.mean(corona_G[1]), np.mean(corona_G[2])]
stdG = [np.std(corona_G[0]), np.std(corona_G[1]), np.std(corona_G[2])]
#print(stdA, stdC, stdT, stdG)
#print(file.split("_")[0] + ' min:' + str(min(ratio_A)) + ' ' + str(min(ratio_C)) + ' ' + str(min(ratio_T)) + ' ' + str(min(ratio_G)))
#print(file.split("_")[0] + ' max:' + str(max(ratio_A)) + ' ' + str(max(ratio_C)) + ' ' + str(max(ratio_T)) + ' ' + str(max(ratio_G)))
labels = ['COVID-19', 'MERS-CoV', 'SARS-CoV']
plt.bar(index,
        dataA,
        yerr=stdA,
        error_kw={'ecolor': '0.1',
                  'capsize': 5
                  },
        alpha=0.7,
        width=0.15,
        label='A',
        color='lightgreen'
        )
plt.bar(index+1*bw, dataC, yerr=stdC, error_kw={'ecolor': '0.1', 'capsize': 5}, alpha=0.7, width=0.15, label='C',
        color='pink')
plt.bar(index+2*bw, dataT, yerr=stdT, error_kw={'ecolor': '0.1', 'capsize': 5}, alpha=0.7, width=0.15, label='T',
        color='blue')
plt.bar(index+3*bw, dataG, yerr=stdG, error_kw={'ecolor': '0.1', 'capsize': 5}, alpha=0.7, width=0.15, label='G',
        color='yellow')
plt.xticks(index+2*bw, labels)
yx = plt.axes()
yx.yaxis.grid()
plt.ylabel('Proportions')
plt.legend(loc='upper center', bbox_to_anchor=(0.48, 1.1), ncol=4)
#plt.show()

plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.savefig('Nucleotide.eps', dpi=500, format='eps')

