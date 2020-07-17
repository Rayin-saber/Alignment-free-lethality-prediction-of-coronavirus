import random


def RealMapping(seq):
    new_seq = []
    for nucleotide in seq:
        if nucleotide == 'A':
            new_seq.append(1.5)
        elif nucleotide == 'G':
            new_seq.append(-0.5)
        elif nucleotide == 'T':
            new_seq.append(-1.5)
        else:
            new_seq.append(0.5)

    return new_seq


def IntegerMapping(seq):
    new_seq = []
    for nucleotide in seq:
        if nucleotide == 'A':
            new_seq.append(0)
        elif nucleotide == 'G':
            new_seq.append(1)
        elif nucleotide == 'T':
            new_seq.append(2)
        else:
            new_seq.append(3)

    return new_seq


def NNMapping(seq):
    new_seq = []
    seq = seq + seq[0]
    for i, nucleotide in enumerate(seq[:-1]):
        if nucleotide + seq[i+1] == 'AA':
            new_seq.append(0)
        elif nucleotide + seq[i+1] == 'AT':
            new_seq.append(1)
        elif nucleotide + seq[i+1] == 'TA':
            new_seq.append(2)
        elif nucleotide + seq[i+1] == 'AG':
            new_seq.append(3)
        elif nucleotide + seq[i+1] == 'TT':
            new_seq.append(4)
        elif nucleotide + seq[i+1] == 'TG':
            new_seq.append(5)
        elif nucleotide + seq[i+1] == 'AC':
            new_seq.append(6)
        elif nucleotide + seq[i+1] == 'TC':
            new_seq.append(7)
        elif nucleotide + seq[i+1] == 'GA':
            new_seq.append(8)
        elif nucleotide + seq[i+1] == 'CA':
            new_seq.append(9)
        elif nucleotide + seq[i+1] == 'GT':
            new_seq.append(10)
        elif nucleotide + seq[i+1] == 'GG':
            new_seq.append(11)
        elif nucleotide + seq[i + 1] == 'CT':
            new_seq.append(12)
        elif nucleotide + seq[i+1] == 'GC':
            new_seq.append(13)
        elif nucleotide + seq[i+1] == 'CG':
            new_seq.append(14)
        else:
            new_seq.append(15)

    return new_seq


def EIIPMapping(seq):
    new_seq = []
    for nucleotide in seq:
        if nucleotide == 'A':
            new_seq.append(0.1260)
        elif nucleotide == 'G':
            new_seq.append(0.0806)
        elif nucleotide == 'T':
            new_seq.append(0.1335)
        else:
            new_seq.append(0.1340)

    return new_seq


def PPMapping(seq):
    new_seq = []
    for nucleotide in seq:
        if nucleotide == 'A':
            new_seq.append(-1)
        elif nucleotide == 'G':
            new_seq.append(-1)
        elif nucleotide == 'T':
            new_seq.append(1)
        else:
            new_seq.append(1)

    return new_seq


def Just_AMapping(seq):
    new_seq = []
    for nucleotide in seq:
        if nucleotide == 'A':
            new_seq.append(1)
        else:
            new_seq.append(0)

    return new_seq


# replace specific symbol in a sequence
def seq_replace(sequence):

    sequence = sequence.replace("N", random.choice(["A", "C", "T", "G"]))
    sequence = sequence.replace("K", random.choice(["T", "G"]))
    sequence = sequence.replace("S", random.choice(["C", "G"]))
    sequence = sequence.replace("W", random.choice(["T", "A"]))
    sequence = sequence.replace("R", random.choice(["A", "G"]))
    sequence = sequence.replace("Y", random.choice(["T", "C"]))
    sequence = sequence.replace("B", random.choice(["T", "G", "C"]))
    sequence = sequence.replace("D", random.choice(["T", "G", "A"]))
    sequence = sequence.replace("H", random.choice(["T", "C", "A"]))
    sequence = sequence.replace("V", random.choice(["C", "G", "A"]))

    return sequence


# ------------------------------------------------------------------------------
# iCGR Encoding: encode a DNA sequence into three integers (iCGR encoding)
# Input: a DNA sequence
# ------------------------------------------------------------------------------
def encodeDNASequence(seq, k):
    lenth = pow(2, k-1)
    A = [lenth, lenth]
    T = [-lenth - 1, lenth]
    C = [-lenth - 1, -lenth - 1]
    G = [lenth, -lenth - 1]
    a = 0
    b = 0
    x = []
    y = []
    n = len(seq)

    if seq[0] == 'A':
        a = 0.5 * A[0]
        b = 0.5 * A[1]
    elif seq[0] == 'T':
        a = 0.5 * T[0]
        b = 0.5 * T[1]
    elif seq[0] == 'C':
        a = 0.5 * C[0]
        b = 0.5 * C[1]
    else:
        a = 0.5 * G[0]
        b = 0.5 * G[1]

    x.append(int(a))
    y.append(int(b))

    for i in range(1, n):
        if seq[i] == 'A':
            a = int((x[i - 1] + A[0]) * 0.5)
            b = int((y[i - 1] + A[1]) * 0.5)
        elif seq[i] == 'T':
            a = int((x[i - 1] + T[0]) * 0.5)
            b = int((y[i - 1] + T[1]) * 0.5)
        elif seq[i] == 'C':
            a = int((x[i - 1] + C[0]) * 0.5)
            b = int((y[i - 1] + C[1]) * 0.5)
        else:
            a = int((x[i - 1] + G[0]) * 0.5)
            b = int((y[i - 1] + G[1]) * 0.5)

        x.append(a)
        y.append(b)
    #print(x)
    #print(y)

    return x, y, n


