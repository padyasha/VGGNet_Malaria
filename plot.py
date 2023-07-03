import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

data_size='3_new'

A = np.loadtxt('result/data_'+data_size+'/val_loss.txt')
B = np.loadtxt('result/data_'+data_size+'/val_accuracy.txt')

C = np.loadtxt('result/data_'+data_size+'/train_loss.txt')
D = np.loadtxt('result/data_'+data_size+'/train_accuracy.txt')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(scipy.signal.savgol_filter(C, 21, 2),label='train')
plt.plot(scipy.signal.savgol_filter(A, 21, 2),label='Val')
plt.legend()
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(scipy.signal.savgol_filter(D, 21, 2),label='train')
plt.plot(scipy.signal.savgol_filter(B, 21, 2),label='Val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()