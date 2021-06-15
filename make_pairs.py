import os
import random
import sys
from itertools import combinations
#input-------------
loot_path = 'path/to/input/directory'
output_path = 'path/to/output/directory'
#------------------
num = 10 
pairs = ''
foldernames = os.listdir(loot_path)
folders_len = len(foldernames)
num_match = 10
diff_max = 0
diff = 0
num_mismatch = 300
f = open(output_path,'w')

for folder in foldernames:
    single_count = 0
    while(True):
        single_filenames = os.listdir(os.path.join(loot_path,folder))
        single_folder_len = len(single_filenames)
        single_file_index = random.randrange(0,single_folder_len)
        single_file_index2 = random.randrange(0,single_folder_len)
        try:
            single_temp = str(folder) + '\t' + str(single_filenames[single_file_index]) + '\t' + str(single_filenames[single_file_index2])
            single_temp_reverse = str(folder) + '\t' + str(single_filenames[single_file_index2]) + '\t' + str(single_filenames[single_file_index])
            #checking duplication
            if(single_temp not in pairs and single_temp_reverse not in pairs and single_file_index != single_file_index2):
                print(single_temp)
                if not pairs:
                    pairs = str(num_match) + '\t' + str(num_mismatch)
                pairs += '\n'+single_temp
                diff_max += 1
                single_count += 1
        except:
            print("error")
            sys.exit()
        if single_count == num or single_count == len(single_filenames)*(len(single_filenames)-1)/2:
            break    

while(True):
    if  diff == diff_max:
        break
    folder_index = random.randrange(0,folders_len)
    folder_index2 = random.randrange(0,folders_len)
    filenames = os.listdir(os.path.join(loot_path,foldernames[folder_index]))
    filenames2 = os.listdir(os.path.join(loot_path,foldernames[folder_index2]))
    folder_len = len(filenames)
    folder2_len = len(filenames2)
    file_index = random.randrange(0,folder_len)
    file_index2 = random.randrange(0,folder2_len)

    try:
        set1 = str(foldernames[folder_index])+'\t'+str(filenames[file_index])
        set2 = str(foldernames[folder_index2])+'\t'+str(filenames2[file_index2])
        temp = set1 + '\t' + set2
        temp_reverse = set2 + '\t' + set2
        #checking duplication
        if(temp not in pairs and temp_reverse not in pairs and foldernames[folder_index] != foldernames[folder_index2]):
            print(temp)
            pairs += '\n'+temp
            diff += 1
    except:
        print("error",file_index2,len(filenames2),foldernames[folder_index2])
        sys.exit()
print("match {} pairs, mismatch {} pairs".format(diff_max,diff))
f.write(pairs)
f.close()    
