
import os
import numpy as np
import cv2


def search(s,mainpath):
    rootdir = mainpath
    searchpath = np.zeros(0)
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.find(s) != -1:
                searchpath = np.append(searchpath, 
                        os.path.abspath(os.path.join(parent,filename)))
    return searchpath

mainpath='data/data_71_new'
ims_list=search('.png',mainpath)

R_means=[]
G_means=[]
B_means=[]
for im_list in ims_list:
	im=cv2.imread(im_list)
#extrect value of diffient channel
	im_R=im[:,:,0]
	im_G=im[:,:,1]
	im_B=im[:,:,2]
#count mean for every channel
	im_R_mean=np.mean(im_R)
	im_G_mean=np.mean(im_G)
	im_B_mean=np.mean(im_B)
#save single mean value to a set of means
	R_means.append(im_R_mean)
	G_means.append(im_G_mean)
	B_means.append(im_B_mean)
#three sets  into a large set
a=[R_means,G_means,B_means]
mean=[0,0,0]
#count the sum of different channel means
mean[0]=np.mean(a[0])
mean[1]=np.mean(a[1])
mean[2]=np.mean(a[2])
print('Means of BGR: \n[{}，{}，{}]'.format( mean[0],mean[1],mean[2]) )