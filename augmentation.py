import os
import numpy as np
import cv2

# cla_list=['artefact','parasite','platelet','rbc']
cla_list=['Healthy','Unhealthy']

for dataset in ['train','val','test']:
        for classes in cla_list:
                ims_path='/home/zelong/Malaria_project/zelong/data/data_256/'+dataset+'/'+classes+'/'
                ims_list=os.listdir(ims_path)
                for im_list in ims_list:
                        im=cv2.imread(ims_path+im_list)
                        for angle in range(2):
                                for flip in range (3):
                                        height, width = im.shape[:2]
                                        M = cv2.getRotationMatrix2D((width/2, height/2), angle*90, 1)
                                        im_rotate=cv2.warpAffine(im, M, (width, height))
                                        im_flip = cv2.flip(im_rotate, flip-1)
                                        filename=ims_path+im_list.split('.')[0]+'_'+str(angle)+'_'+str(flip)+'.png'
                                        cv2.imwrite(filename, im_flip)
                        M = cv2.getRotationMatrix2D((width/2, height/2), 3*90, 1)
                        im_rotate=cv2.warpAffine(im, M, (width, height))
                        im_flip = cv2.flip(im_rotate, -1)
                        filename=ims_path+im_list.split('.')[0]+'_'+str(3)+'_'+str(0)+'.png'
                        cv2.imwrite(filename, im_flip)


# ims_path='/home/zelong/Malaria_project/zelong/data/data_71_4c/test/Unhealthy/'
# ims_list=os.listdir(ims_path)
# for im_list in ims_list:
#     im=cv2.imread(ims_path+im_list)
#     for angle in range(1,4):
#             height, width = im.shape[:2]
#             M = cv2.getRotationMatrix2D((width/2, height/2), angle*90, 1)
#             im_rotate=cv2.warpAffine(im, M, (width, height))
#             filename=ims_path+im_list.split('.')[0]+'_'+str(angle)+'.png'
#             cv2.imwrite(filename, im_rotate)



    