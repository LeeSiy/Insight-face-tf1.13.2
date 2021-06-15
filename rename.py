import os
import pathlib
import argparse
import fnmatch
import re
import random
import shutil

#out_dir = 'path/to/output/directory'
root_dir = 'path/to/input/directory'
#os.mkdir(out_dir+'/train')
#os.mkdir(out_dir+'/valid')
#os.mkdir(out_dir+'/test')

for folder in os.listdir(root_dir):
    folder_dir = os.path.join(root_dir, folder)
    
    for i, name in enumerate(os.listdir(folder_dir)):
        name_dir = os.path.join(folder_dir,name)
        if(i/10<1):
            new_name = folder_dir + "/{}_000{}.jpg".format(folder,i)
        elif(i/100<1):
            new_name = folder_dir + "/{}_00{}.jpg".format(folder,i)
        elif(i/1000<1):
            new_name = folder_dir + "/{}_0{}.jpg".format(folder,i)
        else:
            new_name = folder_dir + "/{}_{}.jpg".format(folder,i)
        os.rename(name_dir, new_name)
    os.rename(folder_dir, root_dir +'/'+ folder)     
    '''
    #os.mkdir(out_dir+"/train/{}".format(folder))
    #os.mkdir(out_dir+"/valid/{}".format(folder))
    os.mkdir(out_dir+"/{}".format(folder))
    files = os.listdir(folder_dir)
    #train_files = random.sample(files,int(len(files)*0.8))
    #for train in train_files:
    #    train_dir = out_dir+"/train/{}".format(folder)
    #    shutil.copyfile(folder_dir+'/'+train,train_dir+'/'+train)
    #    files.remove(train)
    
    #valid_files = random.sample(files,int(len(files)*0.5))
    #for valid in valid_files:
    #    valid_dir = out_dir+"/valid/{}".format(folder)
    #    shutil.copyfile(folder_dir+'/'+valid, valid_dir+'/'+valid)
    #    files.remove(valid)
    files = random.sample(files,int(len(files)*0.1))
    for test in files:
        test_dir = out_dir+"/{}".format(folder)
        shutil.copyfile(folder_dir+'/'+test, test_dir+'/'+test)
    '''
        
        

        
