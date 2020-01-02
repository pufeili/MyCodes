
import os
import random
import shutil
from shutil import copy2

source_dir = "/home/lpf/PycharmProjects/Res71/BreakHis/train/benign"
target_dir = "/home/lpf/PycharmProjects/DataSet/BreakHis64x64/train/benign/"

def data_sampling(source_dir,target_dir,sampleNum):
    all_data = os.listdir(source_dir)  # （图片文件夹）
    num_all_data = len(all_data)
    print("num_all_data: " + str(num_all_data))
    index_list = list(range(num_all_data))
    # print(index_list)
    random.shuffle(index_list)
    # print(index_list)
    num = 0
    for i in index_list:
        fileName = os.path.join(source_dir, all_data[i])
        copy2(fileName, target_dir)
        num += 1
        if (num >= sampleNum):
            break
    return 0

test_benign = data_sampling(source_dir,target_dir,sampleNum=30000)













