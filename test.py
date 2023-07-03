import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from vgg16 import VGG16
from datagenerator import ImageDataGenerator
import cv2

"""
Configuration settings
"""

# data params
data_size='3_new'

# Learning params
num_classes=2
batch_size = 1

train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2','conv1']
# train_layers = ['fc8', 'fc7','fc6']

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = VGG16(x, keep_prob, num_classes, train_layers)


# Initialize an saver for store model checkpoints
saver = tf.train.Saver()


if data_size == '256': rgb_mean = 127.5093
elif data_size == '128': rgb_mean = 127.0391
elif data_size == '71': rgb_mean = 127.0901
elif data_size == '71_new': rgb_mean = 124.9396
elif data_size == '1_new': rgb_mean = 32.1265
elif data_size == '2_new': rgb_mean = 30.2056
elif data_size == '3_new': rgb_mean = 11.9783
elif data_size == '4_new': rgb_mean = 26.5960

def imrd(path):
    images = np.ndarray([1, 224, 224, 3])
    img= cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = img.astype(np.float32)
    img = img-rgb_mean
    images[0] = img
    return images

class_list = ['Healthy','Unhealthy']
starttime=datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'result/data_'+data_size+'/checkpoint/model_epoch20.ckpt')
    acc=np.zeros(2)
    test_acc=np.zeros(2)
    test_count=np.zeros(2)
    labels = []
    pred = []
    for idx in range(2):
        mainpath='data/data_'+data_size+'/test/'+class_list[idx]
        for file in os.listdir(mainpath):
            images=imrd(mainpath+'/'+file)
            score = sess.run(model.fc8, feed_dict={x:images, keep_prob:1.0})
            result = np.exp(score)/np.sum(np.exp(score))
            labels = np.r_[labels,int(idx)]
            pred = np.r_[pred,result[0,1]]
            if np.max(result)==result[0,idx]:
                test_acc[idx]+=1
            test_count[idx]+=1
        acc[idx] = test_acc[idx]/test_count[idx]
        print("{} test Accuracy of {} = {:.4f}%  [{:d}/{:d}]".format(datetime.now(),class_list[idx], acc[idx]*100,int(test_acc[idx]),int(test_count[idx])))
    print("{} test Accuracy = {:.4f}%  [{:d}/{:d}]".format(datetime.now(), (test_acc[0]+test_acc[1])*100/(test_count[0]+test_count[1]),int((test_acc[0]+test_acc[1])),int((test_count[0]+test_count[1]))))
    print('F1 score: ', 2*test_acc[1]/(test_acc[1]+test_count[0]-test_acc[0]+test_count[1]))
    np.savetxt('result/data_'+data_size+'/labels.txt',labels,fmt='%d')
    np.savetxt('result/data_'+data_size+'/pred.txt',pred)

endtime=datetime.now()
print('Test time: ',(endtime-starttime).seconds)