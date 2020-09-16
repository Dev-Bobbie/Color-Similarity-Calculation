import cv2
import math
from math import sqrt
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cmc
from PIL import Image
import time
import random
import os
import sys
sys.path.append('..')
change_scale=1
class Hash():
    # 差异值哈希算法
    def dhash(self,image):
        height, width, channels = image.shape
        resize_height,resized_width=int(height*change_scale), int(width*change_scale)
        # resize_height, resized_width = 32, 32
        # 缩放到(resized_width, resize_height)尺寸的大小

        resized_img = cv2.resize(image, (resized_width, resize_height))
        # 图片灰度化
        grey_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        # 差异值计算
        hash_list = []
        for row in range(resize_height):
            for col in range(resized_width - 1):
                # 每行前一个颜色强度大于后一个，值为1，否则值为0
                if grey_resized_img[row, col] > grey_resized_img[row, col + 1]:
                    hash_list.append('1')
                else:
                    hash_list.append('0')

        return '' . join(hash_list)
# 比较汉明距离
    def hamming_distance(self,dhash1, dhash2):
        return bin(int(dhash1, base = 2) ^ int(dhash2, base = 2)).count('1')
    def color_test(self,img1_path,img2_path):
        # 读取图片内容
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        return self.hamming_distance(self.dhash(img1),self.dhash(img2))
