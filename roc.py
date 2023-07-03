import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt


# data_size=128
data_size=['1_new','2_new','3_new','4_new']
plt.figure()
for idx in range(4):
    label=np.loadtxt('result/data_'+data_size[idx]+'/labels.txt')
    pred=np.loadtxt('result/data_'+data_size[idx]+'/pred.txt')
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    auc=metrics.roc_auc_score(label,pred)

    plt.plot(fpr,tpr,label='ROC curve of data_%d (auc = %0.4f)' %(idx+1, auc))
plt.plot([0,1],[0,1],'k--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of VGGNet')
plt.show()