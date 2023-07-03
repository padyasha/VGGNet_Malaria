import os



for dataset in ['train','val','test']:
    path='/home/zelong/Malaria_project/zelong/data/data_71_4c/'+dataset+'/artefact'
    fileObject = open('datatxt/'+dataset+'_71_4c.txt', 'a+')
    for file in os.listdir(path):
        combine = path+'/'+file+ ' '+'0'
        fileObject.write(combine)
        fileObject.write('\n')
    fileObject.close()
    path='/home/zelong/Malaria_project/zelong/data/data_71_4c/'+dataset+'/parasite'
    fileObject = open('datatxt/'+dataset+'_71_4c.txt', 'a+')
    for file in os.listdir(path):
        combine = path+'/'+file+ ' '+'1'
        fileObject.write(combine)
        fileObject.write('\n')
    fileObject.close()
    path='/home/zelong/Malaria_project/zelong/data/data_71_4c/'+dataset+'/platelet'
    fileObject = open('datatxt/'+dataset+'_71_4c.txt', 'a+')
    for file in os.listdir(path):
        combine = path+'/'+file+ ' '+'2'
        fileObject.write(combine)
        fileObject.write('\n')
    fileObject.close()
    path='/home/zelong/Malaria_project/zelong/data/data_71_4c/'+dataset+'/rbc'
    fileObject = open('datatxt/'+dataset+'_71_4c.txt', 'a+')
    for file in os.listdir(path):
        combine = path+'/'+file+ ' '+'3'
        fileObject.write(combine)
        fileObject.write('\n')
    fileObject.close()

