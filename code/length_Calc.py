import numpy as np
import os
import CGR

def length_calc(path):
    length = []
    files = os.listdir(path)
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
                    DNASeq = CGR.seq_replace(sequence)
                    length.append(len(DNASeq))
                    print(len(DNASeq))

    medLen = np.median(np.array(length))  # return the median length of all sequences

    return int(medLen)

if __name__ == '__main__':

    length_calc("E:/2019 NTU/Project/Program/Corona Virus/data/nucleotide/subtype/TEST")