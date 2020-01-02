

import os
from shutil import copy2

source_dir = '/home/li/LPF/sq/代码final/BK/testBK/'
target_dir = '/home/li/LPF/ResNet/BreakH/test/'

def moveImage(source_dir,target_dir):
    filenames = os.listdir(source_dir)
    print("original imgs is:",len(filenames))
    num = 0

    for item in filenames:
        num += 1
        img = os.path.join(source_dir,item)
        if item[0] == 'C':
            copy2(img,os.path.join(target_dir,'1'))
        elif item[0] == 'A':
            copy2(img, os.path.join(target_dir,'0'))
    print("copy num is: %d"%num)

aa = moveImage(source_dir,target_dir)








