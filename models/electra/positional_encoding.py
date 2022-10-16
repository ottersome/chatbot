import numpy as np
import matplotlib as plt

dimensions =4
d = dimensions
sequence_length = 4
n = 100

EncodingMatrix = np.zeros((sequence_length,dimensions))
for idx in np.arange(4):

    b0 = np.arange(0,int(d/2))# Creates range
    b0 = np.repeat(b0,2)
    b1 = idx/np.power(n,2*b0/d)
    b1[0::2] = np.sin(b1[0::2])
    b1[1::2] = np.cos(b1[0::2])
    EncodingMatrix[idx,:] = b1

print("Encoding Matrix is : ")
print(EncodingMatrix)


