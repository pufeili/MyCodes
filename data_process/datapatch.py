import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time

'''
*************************
  注意：图片的width和height 
**************************
#当crop()中的参数超出了图片的界限时，用0填充
#upper_left_x：与左边界的距离
#upper_left_y：与上边界的距离
#lower_right_x：与左边界的距离
#lower_right_y：与上边界的距离
'''
# read_path = '/home/lpf/PycharmProjects/Res71/BreaKHis_v1/train/benign'
# save_path = '/home/lpf/PycharmProjects/Res71/BreakHis32X32/train/benign/'

# read_path = '/home/lpf/PycharmProjects/Res71/Test/malignant'
# save_path = '/home/lpf/PycharmProjects/Res71/Test/qq/'

file_list = os.listdir(read_path)
number_file = len(file_list)
print("number of files is: %d"%number_file)
number = 0
generating = 0
for file in (file_list):
    a,b = os.path.splitext(file)
    if (b == '.png'):
        im = Image.open(os.path.join(read_path+ "/" + file))
        width,hight = im.size
        # print(width,hight)
        size = 32 #size
        l = 30    #step
        id = 1
        i = 0
        while (i + size <= width):
            j = 0
            while (j + size <= hight):
                new_img = im.crop((i, j, i + size, j + size)) #(upper_left_x,upper_left_y,lower_right_x,lower_right_y)
                # print(type(new_img))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                new_img.save(save_path + a + "_" + str(id) + '.png')
                id += 1
                j += l
                generating +=1
            # print(j)
            i = i + l
    number += 1
    if (number%10 == 0):
        print("patching: [%d / %d] ,total genertating pic: %d"%(number,number_file,generating))
        time.sleep(0.01)


#upper_left_x：与左边界的距离
#upper_left_y：与上边界的距离
#lower_right_x：与左边界的距离
#lower_right_y：与上边界的距离













