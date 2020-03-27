import cv2
import math
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
import os
from operator import itemgetter
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cmc
from PIL import Image
import time
import random
import sys
from torchvision import transforms as tfs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
sys.path.append('..')
change_scale=1
class CDH():
    def calc_hist(self,imagepath):
       
        img = cv2.imread(imagepath)
        resize_height,resized_width=32,32
        img = cv2.resize(img, (resize_height,resized_width))

        # height, width, channels = img.shape
        # resize_height,resized_width=int(height*change_scale), int(width*change_scale)
        # img = cv2.resize(img, (resize_height,resized_width))


        width, height, channels = img.shape
        lnum = 10
        anum = 3
        bnum = 3
        cnum = lnum * anum * bnum
        onum = 18
        hist = np.zeros(cnum + onum)
        QuantizedImage = np.zeros(width * height).reshape(width, height)
        L = a = b = 0
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for i in range(width):
            for j in range(height):
                L = int(Lab[i][j][0] * lnum / 100.0)
                if L >= lnum - 1:
                    L = lnum - 1
                elif L < 0:
                    L = 0
                
                a = int(Lab[i][j][1] * anum / 254.0)
                if a >= anum - 1:
                    a = anum - 1
                elif a < 0:
                    a = 0
                    
                b = int(Lab[i][j][2] * bnum / 254.0)
                if b >= anum - 1:
                    b = anum - 1
                elif b < 0:
                    b = 0
                    
                QuantizedImage[i][j] = bnum * anum * L + bnum * a + b
        
         # In[13]:
        lab = self.coOrdinateTransform(Lab, width, height)
        ori = self.maxgrad_and_mingrad_Lab(lab, onum, width, height)
        hist = self.compute(QuantizedImage,ori,Lab,width,height,cnum,onum,1)
        return hist
    # In[12]:
    def coOrdinateTransform(self,arr, width, height):
        Lab = np.zeros(3 * width * height).reshape(width, height, 3)
        for i in range(width):
            for j in range(height):
                Lab[i][j][0] = arr[i][j][0]
                Lab[i][j][1] = arr[i][j][1]
                Lab[i][j][2] = arr[i][j][2]
                
                Lab[i][j][1] = arr[i][j][1] + 127.0
                if Lab[i][j][1] >= 254.0:
                    Lab[i][j][1] = 254.0 - 1.0
                if Lab[i][j][1] < 0:
                    Lab[i][j][1] = 0
                
                Lab[i][j][2] = arr[i][j][2] + 127.0
                if Lab[i][j][2] >= 254.0:
                    Lab[i][j][2] = 254.0 - 1.0
                if Lab[i][j][2] < 0:
                    Lab[i][j][2] = 0
        return Lab

   

    # In[14]:
    def maxgrad_and_mingrad_Lab(self,arr,num,wid,hei):
        gxx = gyy = gxy = 0.0
        rh = gh = bh = 0.0
        rv = gv = bv = 0.0
        theta = 0.0
        ori = np.zeros(wid * hei).reshape(wid, hei)
        for i in range(1, wid - 1):
            for j in range(1, hei - 1):
                rh=arr[i-1,j+1,0] + 2*arr[i,j + 1,0] + arr[i+1, j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i,j-1, 0] + arr[i + 1, j - 1, 0])
                gh=arr[i-1,j+1,1] + 2*arr[i,j + 1,1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i,j-1, 1] + arr[i + 1, j - 1, 1])
                bh=arr[i-1,j+1,2] + 2*arr[i,j + 1,2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i,j-1, 2] + arr[i + 1, j - 1, 2])
                rv=arr[i+1,j-1,0] + 2*arr[i+1, j, 0] + arr[i+ 1,j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i-1,j, 0] + arr[i - 1, j + 1, 0])
                gv=arr[i+1,j-1,1] + 2*arr[i+1, j, 1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i-1,j, 1] + arr[i - 1, j + 1, 1])
                bv=arr[i+1,j-1,2] + 2*arr[i+1, j, 2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i-1,j, 2] + arr[i - 1, j + 1, 2])

                gxx = rh * rh + gh * gh + bh * bh
                gyy = rv * rv + gv * gv + bv * bv
                gxy = rh * rv + gh * gv + bh * bv

                theta = round(math.atan(2.0 * gxy / (gxx - gyy + 0.00001))/ 2.0, 4)
                G1 = G2 = 0.0

                G1 = math.sqrt(abs(0.5 * ((gxx + gyy) + (gxx - gyy) * math.cos(2.0 * theta) + 2.0 * gxy * math.sin(2.0 * theta))))
                G2=math.sqrt(abs(0.5*((gxx+gyy)+(gxx-gyy)*math.cos(2.0*(theta +(math.pi/2.0)))+ 2.0 * gxy * math.sin(2.0*(theta+ (math.pi / 2.0))))))

                dir = 0

                if max(G1, G2) == G1:
                    dir = 90.0 + theta * 180.0 / math.pi
                    ori[i, j] = int(dir * num / 360.0)
                else:
                    dir = 180.0 + (theta + math.pi / 2.0) * 180.0 / math.pi
                    ori[i, j] = int(dir * num / 360.0)
                if ori[i, j] >= num - 1:
                    ori[i, j] = num - 1
        return ori


    # In[15]:

    

    # In[16]:
    def compute(self,ColorX,ori,Lab,wid,hei,CSA,CSB,D):
        Arr = np.zeros(3 * wid * hei).reshape(wid, hei, 3)
        Arr = self.coOrdinateTransform(Lab, wid, hei)
        Matrix = np.zeros(CSA + CSB).reshape(CSA + CSB)
        hist = np.zeros(CSA + CSB).reshape(CSA + CSB)

        # -------------------calculate the color difference of different directions------------

        # ----------direction=0--------------------

        for i in range(wid):
            for j in range(hei - D):
                value = 0.0
                if ori[i, j + D] == ori[i, j]:
                    value = math.sqrt(math.pow(Arr[i, j + D, 0] - Arr[i, j,0], 2) + math.pow(Arr[i, j + D, 1]- Arr[i, j, 1], 2) + math.pow(Arr[i,j + D, 2] - Arr[i, j, 2], 2))
                    Matrix[int(ColorX[i, j])] += value
                if ColorX[i, j + D] == ColorX[i, j]:
                    value = math.sqrt(math.pow(Arr[i, j + D, 0] - Arr[i, j,0], 2) + math.pow(Arr[i, j + D, 1]- Arr[i, j, 1], 2) + math.pow(Arr[i,j + D, 2] - Arr[i, j, 2], 2))
                    Matrix[int(ori[i, j] + CSA)] += value

            # -----------direction=90---------------------
        
        for i in range(wid - D):
            for j in range(hei):
                value = 0.0
                if ori[i + D, j] == ori[i, j]:
                    value = math.sqrt(math.pow(Arr[i + D, j, 0] - Arr[i, j,0], 2) + math.pow(Arr[i + D, j, 1]- Arr[i, j, 1], 2) + math.pow(Arr[i+ D, j, 2] - Arr[i, j, 2], 2))
                    Matrix[int(ColorX[i, j])] += value
                if ColorX[i + D, j] == ColorX[i, j]:
                    value = math.sqrt(math.pow(Arr[i + D, j, 0] - Arr[i, j,0], 2) + math.pow(Arr[i + D, j, 1]- Arr[i, j, 1], 2) + math.pow(Arr[i+ D, j, 2] - Arr[i, j, 2], 2))
                    Matrix[int(ori[i, j] + CSA)] += value

        # -----------direction=135---------------------
        for i in range(wid - D):
            for j in range(hei - D):
                value = 0.0
                if ori[i + D, j + D] == ori[i, j]:
                    value = math.sqrt(math.pow(Arr[i + D, j + D, 0]- Arr[i, j, 0], 2) + math.pow(Arr[i+ D, j + D, 1] - Arr[i, j, 1], 2)+ math.pow(Arr[i + D, j + D, 2]- Arr[i, j, 2], 2))
                    Matrix[int(ColorX[i, j])] += value
                if ColorX[i + D, j + D] == ColorX[i, j]:
                    value = math.sqrt(math.pow(Arr[i + D, j + D, 0]- Arr[i, j, 0], 2) + math.pow(Arr[i+ D, j + D, 1] - Arr[i, j, 1], 2)+ math.pow(Arr[i + D, j + D, 2]- Arr[i, j, 2], 2))
                    Matrix[int(ori[i, j] + CSA)] += value
                
        # -----------direction=45---------------------
        
        for i in range(D, wid):
            for j in range(hei - D):
                value = 0.0
                if ori[i - D, j + D] == ori[i, j]:
                    value = math.sqrt(math.pow(Arr[i - D, j + D, 0]- Arr[i, j, 0], 2) + math.pow(Arr[i- D, j + D, 1] - Arr[i, j, 1], 2)+ math.pow(Arr[i - D, j + D, 2]- Arr[i, j, 2], 2))
                    Matrix[int(ColorX[i, j])] += value
                if ColorX[i - D, j + D] == ColorX[i, j]:
                    value = math.sqrt(math.pow(Arr[i - D, j + D, 0]- Arr[i, j, 0], 2) + math.pow(Arr[i- D, j + D, 1] - Arr[i, j, 1], 2)+ math.pow(Arr[i - D, j + D, 2]- Arr[i, j, 2], 2))
                    Matrix[int(ori[i, j] + CSA)] += value
                    
        for i in range(CSA + CSB):
            hist[i] = (Matrix[i]) / 4.0
            
        return hist


    # def compare_similar_hist(self,h1, h2):
    #     sum1, sum2, sum_mixd = 0, 0, 0
    #     # 像素值key的最大数不超过512，直接循环到512，遍历取出每个像素值
    #     for i in range(108):
    #         # 计算出现相同像素值次数的平方和
    #         sum1 = sum1 + h1[i] * h1[i]
    #         sum2 = sum2 + h2[i] * h2[i]
    #         # 计算两个图片次数乘积的和
    #         sum_mixd = sum_mixd + (h1[i] * h2[i])
    #     # 按照余弦相似性定理计算相似度
    #     return sum_mixd / (sqrt(sum1) * sqrt(sum2))
    
    def compare_similar_hist(self,h1, h2):
        sum = 0
        # 像素值key的最大数不超过512，直接循环到512，遍历取出每个像素值
        for i in range(len(h1)):
            # 计算出现相同像素值次数的平方和
            sum = sum + abs(h1[i] - h2[i])/(1+h1[i] + h2[i])
        # 按照余弦相似性定理计算相似度
        return -sum
    def color_test(self,srcpath,dstpath):
        end_ans=self.compare_similar_hist(self.calc_hist(srcpath), self.calc_hist(dstpath))
        return end_ans


class MTH():
    def calc_hist(self,imagepath):
        img = cv2.imread(imagepath)
        resize_height,resized_width=32,32
        img = cv2.resize(img, (resize_height,resized_width))

        # height,width, channels = img.shape
        # resize_height,resized_width=int(height*change_scale), int(width*change_scale)
        # img = cv2.resize(img, (resize_height,resized_width)

        width, height, channels = img.shape
        # # Texture Orientation Detection
        CSA = 64
        CSB = 18
        arr = np.zeros(3*width*height).reshape(width,height,3)
        ori = np.zeros(width * height).reshape(width, height)

        # In[9]:

        gxx = gyy = gxy = 0.0
        rh = gh = bh = 0.0
        rv = gv = bv = 0.0
        theta = np.zeros(width*height).reshape(width,height)

        for i in range(1, width-2):
            for j in range(1, height-2):
                rh=arr[i-1,j+1,0] + 2*arr[i,j + 1,0] + arr[i+1, j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i,j-1, 0] + arr[i + 1, j - 1, 0])
                gh=arr[i-1,j+1,1] + 2*arr[i,j + 1,1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i,j-1, 1] + arr[i + 1, j - 1, 1])
                bh=arr[i-1,j+1,2] + 2*arr[i,j + 1,2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i,j-1, 2] + arr[i + 1, j - 1, 2])
                rv=arr[i+1,j-1,0] + 2*arr[i+1, j, 0] + arr[i+ 1,j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i-1,j, 0] + arr[i - 1, j + 1, 0])
                gv=arr[i+1,j-1,1] + 2*arr[i+1, j, 1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i-1,j, 1] + arr[i - 1, j + 1, 1])
                bv=arr[i+1,j-1,2] + 2*arr[i+1, j, 2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i-1,j, 2] + arr[i - 1, j + 1, 2])
                
                gxx = math.sqrt(rh * rh + gh * gh + bh * bh)
                gyy = math.sqrt(rv * rv + gv * gv + bv * bv)
                gxy = rh * rv + gh * gv + bh * bv
                
                theta[i,j] = (math.acos(gxy / (gxx * gyy + 0.0001))*180 / math.pi)

        ImageX = np.zeros(width * height).reshape(width, height)

        R = G = B = 0
        VI = SI = HI = 0
        for i in range(0, width):
            for j in range(0, height):
                R = img[i,j][0]
                G = img[i,j][1]
                B = img[i,j][2]
                
                if (R >=0 and R <= 64):
                    VI = 0
                if (R >= 65 and R <= 128):
                    VI = 1
                if (R >= 129 and R <= 192):
                    VI = 2
                if (R >= 193 and R <= 255):
                    VI = 3
                if (G>= 0 and G <= 64):
                    SI = 0
                if (G >= 65 and G <= 128):
                    SI = 1
                if (G >= 129 and G <= 192):
                    SI = 2
                if (G >= 193 and G <= 255):
                    SI = 3
                if (B >= 0 and B <= 64):
                    HI = 0
                if (B >= 65 and B <= 128):
                    HI = 1
                if (B >= 129 and B <= 192):
                    HI = 2
                if (B >= 193 and B <= 255):
                    HI = 3
                
                ImageX[i, j] = 16 * VI + 4 * SI + HI


        # In[11]:


        for i in range(0, width):
            for j in range(0, height):
                ori[i,j] = round(theta[i,j]*CSB/180)
                
                if(ori[i,j]>=CSB-1):
                    ori[i,j]=CSB-1
  

        Texton = np.zeros(width * height).reshape(width, height)

        for i in range(0,(int)(width/2)):
            for j in range(0,(int)(height/2)):
                if(ImageX[2*i,2*j] == ImageX[2*i+1,2*j+1]):
                    Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j]
                    Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j]
                    Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1]
                    Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1]
                
                if (ImageX[2*i,2*j+1] == ImageX[2*i+1,2*j]):
                    Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j]
                    Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j]
                    Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1]
                    Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1]
                
                if (ImageX[2*i,2*j] == ImageX[2*i+1,2*j]): 
                    Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j]
                    Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j]
                    Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1]
                    Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1]
                    
                if (ImageX[2*i,2*j] == ImageX[2*i,2*j+1]):
                    Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j]
                    Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j]
                    Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1]
                    Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1]                   


        # # Multi-Texton Histogram

        # In[13]:


        MatrixH = np.zeros(CSA + CSB).reshape(CSA + CSB)
        MatrixV = np.zeros(CSA + CSB).reshape(CSA + CSB)
        MatrixRD = np.zeros(CSA + CSB).reshape(CSA + CSB)
        MatrixLD = np.zeros(CSA + CSB).reshape(CSA + CSB)

        D = 1 #distance parameter

        for i in range(0, width):
            for j in range(0, height-D):
                if(ori[i, j+D] == ori[i, j]):
                    MatrixH[(int)(Texton[i,j])] += 1
                if(Texton[i, j + D] == Texton[i, j]):
                    MatrixH[(int)(CSA + ori[i, j])] += 1

        for i in range(0, width-D):
            for j in range(0, height):
                if(ori[i + D, j] == ori[i, j]):
                    MatrixV[(int)(Texton[i,j])] += 1
                if(Texton[i + D, j] == Texton[i, j]):
                    MatrixV[(int)(CSA + ori[i, j])] += 1

        for i in range(0, width-D):
            for j in range(0, height-D):
                if(ori[i + D, j + D] == ori[i, j]):
                    MatrixRD[(int)(Texton[i,j])] += 1
                if(Texton[i + D, j + D] == Texton[i, j]):
                    MatrixRD[(int)(CSA + ori[i, j])] += 1
                    
        for i in range(D, width):
            for j in range(0, height-D):
                if(ori[i - D, j + D] == ori[i, j]):
                    MatrixLD[(int)(Texton[i,j])] += 1
                if(Texton[i - D, j + D] == Texton[i, j]):
                    MatrixLD[(int)(CSA + ori[i, j])] += 1

        # # Feature Vectors

        # In[14]:


        MTH = np.zeros(CSA + CSB).reshape(CSA + CSB)

        for i in range(0, CSA + CSB):
            MTH[i] = ( MatrixH[i] + MatrixV[i] + MatrixRD[i] + MatrixLD[i])/4.0
        return MTH
    # def compare_similar_hist(self,h1, h2):
    #     sum1, sum2, sum_mixd = 0, 0, 0
    #     # 像素值key的最大数不超过512，直接循环到512，遍历取出每个像素值
    #     for i in range(len(h1)):
    #         # 计算出现相同像素值次数的平方和
    #         sum1 = sum1 + h1[i] * h1[i]
    #         sum2 = sum2 + h2[i] * h2[i]
    #         # 计算两个图片次数乘积的和
    #         sum_mixd = sum_mixd + (h1[i] * h2[i])
    #     # 按照余弦相似性定理计算相似度
    #     return sum_mixd / (sqrt(sum1) * sqrt(sum2))
    def compare_similar_hist(self,h1, h2):
        sum = 0
        # 像素值key的最大数不超过512，直接循环到512，遍历取出每个像素值
        for i in range(len(h1)):
            # 计算出现相同像素值次数的平方和
            sum = sum + abs(h1[i] - h2[i])/(1+h1[i] + h2[i])
        # 按照余弦相似性定理计算相似度
        return -sum
    def color_test(self,srcpath,dstpath):
        end_ans=self.compare_similar_hist(self.calc_hist(srcpath), self.calc_hist(dstpath))
        return end_ans


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

class Histogram():
    # 颜色映射
    def bgr_mapping(self,img_val):
    # 将bgr颜色分成8个区间做映射
        if (img_val >= 0 and img_val <= 31): return 0
        if img_val >= 32 and img_val <= 63: return 1
        if img_val >= 64 and img_val <= 95: return 2
        if img_val >= 96 and img_val <= 127: return 3
        if img_val >= 128 and img_val <= 159: return 4
        if img_val >= 160 and img_val <= 191: return 5
        if img_val >= 192 and img_val <= 223: return 6
        if img_val >= 224: return 7

    # 颜色直方图的数值计算
    def calc_bgr_hist(self,image):
        if not image.size: return False
        hist = {}
        # 缩放尺寸减小计算量
        
        height, width, channels = image.shape
        resize_height,resized_width=int(height*change_scale), int(width*change_scale)
        # resize_height,resized_width=32,32
        image = cv2.resize(image, (resize_height,resized_width))
        for bgr_list in image:
            for bgr in bgr_list:
                # 颜色按照顺序映射
                maped_b = self.bgr_mapping(bgr[0])
                maped_g = self.bgr_mapping(bgr[1])
                maped_r = self.bgr_mapping(bgr[2])
                # 计算像素值
                index   = maped_b * 8 * 8 + maped_g * 8 + maped_r
                hist[index] = hist.get(index, 0) + 1

        return hist

    def compare_similar_hist(self,h1, h2):
        if not h1 or not h2: return False
        sum1, sum2, sum_mixd = 0, 0, 0
        # 像素值key的最大数不超过512，直接循环到512，遍历取出每个像素值
        for i in range(512):
            # 计算出现相同像素值次数的平方和
            sum1 = sum1 + (h1.get(i, 0) * h1.get(i, 0))
            sum2 = sum2 + (h2.get(i, 0) * h2.get(i, 0))
            # 计算两个图片次数乘积的和
            sum_mixd = sum_mixd + (h1.get(i, 0) * h2.get(i, 0))
        # 按照余弦相似性定理计算相似度
        return sum_mixd / (sqrt(sum1) * sqrt(sum2))
    def color_test(self,img1_path,img2_path):
        img1 = cv2.imread(img1_path)
        # 读取图片内容
        img2 = cv2.imread(img2_path)
        return self.compare_similar_hist(self.calc_bgr_hist(img1), self.calc_bgr_hist(img2))

class Color():
    def __init__(self):
            # region 辅助函数
    # RGB2XYZ空间的系数矩阵
        self.M = np.array([[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]])
#区域划分
    def bgr_mapping(self,img_val):
        # 将bgr颜色分成256/64个区间做映射
        return int(img_val/64)
# RGb的加权数值计算
    def calc_bgr_hist(self,im):
        # 缩放尺寸减小计算量
        
        # im = im.resize((32, 32), Image.ANTIALIAS)
        # width,height= 32,32
        width,height= int(im.size[0]*change_scale),int(im.size[1]*change_scale)
        im = im.resize((width, height), Image.ANTIALIAS)

        pix = im.load()
        hist1={}; hist2={}; hist3={}
        L_width,R_width=int(0.1*width),int(0.9*width)
        L_height,R_height=int(0.1*height),int(0.9*height)
        for x in range(L_width,R_width):
            for y in range(L_height,R_height):
                if(len(pix[x,y])==3):
                    maped_r, maped_g, maped_b= pix[x, y]
                else:
                    maped_r, maped_g, maped_b ,appra= pix[x, y]
                    if(appra==0):
                        continue
                # 计算像素值
                maped_b = self.bgr_mapping(maped_b)
                maped_g = self.bgr_mapping(maped_g)
                maped_r = self.bgr_mapping(maped_r)
                hist1[maped_b] = hist1.get(maped_b, 0) + 1
                hist2[maped_g] = hist2.get(maped_g, 0) + 1
                hist3[maped_r] = hist3.get(maped_r, 0) + 1
        B_L=sorted(hist1.items(),key=itemgetter(1),reverse=True)
        G_L=sorted(hist2.items(),key=itemgetter(1),reverse=True)
        R_L=sorted(hist3.items(),key=itemgetter(1),reverse=True)

        for i in range(len(B_L),3):
            B_L.append([0,0])    
        B_mean = (B_L[0][0]*B_L[0][1]+B_L[1][0]*B_L[1][1]+B_L[2][0]*B_L[2][1])/(1.0*B_L[0][1]+B_L[1][1]+B_L[2][1])

        for i in range(len(G_L),3):
            G_L.append([0,0])    
        G_mean = (G_L[0][0]*G_L[0][1]+G_L[1][0]*G_L[1][1]+G_L[2][0]*G_L[2][1])/(1.0*G_L[0][1]+G_L[1][1]+G_L[2][1])

        for i in range(len(R_L),3):
            R_L.append([0,0])    
        R_mean = (R_L[0][0]*R_L[0][1]+R_L[1][0]*R_L[1][1]+R_L[2][0]*R_L[2][1])/(1.0*R_L[0][1]+R_L[1][1]+R_L[2][1])

        return [B_mean*64,G_mean*64,R_mean*64]
    # im_channel取值范围：[0,1]
    def f(self,im_channel):
        return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931
    def anti_f(self,im_channel):
        return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
    # endregion
    # region RGB 转 Lab
    # 像素值RGB转XYZ空间，pixel格式:(B,G,R)
    # 返回XYZ空间下的值
    def __rgb2xyz__(self,pixel):
        b, g, r = pixel[0], pixel[1], pixel[2]
        rgb = np.array([r, g, b])
        # rgb = rgb / 255.0
        # RGB = np.array([gamma(c) for c in rgb])
        XYZ = np.dot(self.M, rgb.T)
        XYZ = XYZ / 255.0
        return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)
    def __xyz2lab__(self,xyz):
        """
        XYZ空间转Lab空间
        :param xyz: 像素xyz空间下的值
        :return: 返回Lab空间下的值
        """
        F_XYZ = [self.f(x) for x in xyz]
        L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
        a = 500 * (F_XYZ[0] - F_XYZ[1])
        b = 200 * (F_XYZ[1] - F_XYZ[2])
        return (L, a, b)


    def RGB2Lab(self,pixel):
        """
        RGB空间转Lab空间
        :param pixel: RGB空间像素值，格式：[G,B,R]
        :return: 返回Lab空间下的值
        """
        xyz = self.__rgb2xyz__(pixel)
        Lab = self.__xyz2lab__(xyz)
        return Lab


    # endregion

    # region Lab 转 RGB
    def __lab2xyz__(self,Lab):
        fY = (Lab[0] + 16.0) / 116.0
        fX = Lab[1] / 500.0 + fY
        fZ = fY - Lab[2] / 200.0

        x = self.anti_f(fX)
        y = self.anti_f(fY)
        z = self.anti_f(fZ)

        x = x * 0.95047
        y = y * 1.0
        z = z * 1.0883

        return (x, y, z)


    def __xyz2rgb(self,xyz):
        xyz = np.array(xyz)
        xyz = xyz * 255
        rgb = np.dot(np.linalg.inv(self.M), xyz.T)
        # rgb = rgb * 255
        rgb = np.uint8(np.clip(rgb, 0, 255))
        return rgb


    def Lab2RGB(self,Lab):
        xyz = self.__lab2xyz__(Lab)
        rgb = self.__xyz2rgb(xyz)
        return rgb
    # endregion

    #CIEDE
    def compare_similar_hist(self,rgb_1, rgb_2):
        lab_l,lab_a,lab_b=self.RGB2Lab(rgb_1)
        color1 = LabColor(lab_l, lab_a, lab_b)

        lab_l,lab_a,lab_b=self.RGB2Lab(rgb_2)
        color2 = LabColor(lab_l, lab_a, lab_b)
        delta_e = delta_e_cie2000(color1, color2)
        # delta_e = delta_e_cmc(color1, color2)
        return delta_e
    # 读取图片内容
    def color_test(self,srcpath,dstpath):
        im1 = Image.open(srcpath)
        im2 = Image.open(dstpath)
        end_ans=self.compare_similar_hist(self.calc_bgr_hist(im1), self.calc_bgr_hist(im2))
        return end_ans


class Dataset():     
    set_size=90
    batch_size=1
    def __init__(self,set_size=90,batch_size=1):
        self.imageFolderDataset = []
        self.train_dataloader = []
        self.__set_size__ = set_size
        self.__batch_size__ =batch_size
    def __getitem__(self, class_num=40):
        '''
        如果图像来自同一个类，标签将为0，否则为1
        Y值为1或0。如果模型预测输入是相似的，那么Y的值为0，否则Y为1。
        '''
        should_get_same_class = random.randint(0,1)
        for i in range(self.__batch_size__):
            img0_class = random.randint(0,class_num-1)
            # we need to make sure approx 50% of images are in the same class
            item_num = len(self.imageFolderDataset[img0_class])
            temp = random.sample(list(range(0,item_num)), 2)
            img0_tuple = (self.imageFolderDataset[img0_class][temp[0]], img0_class)
            img1_tuple = (self.imageFolderDataset[img0_class][temp[1]], img0_class)
            img2_class = random.randint(0, class_num - 1)
            # 保证属于不同类别
            while img2_class == img0_class:
                img2_class = random.randint(0, class_num - 1)
            item_num = len(self.imageFolderDataset[img2_class])
            img2_tuple = (self.imageFolderDataset[img2_class][random.randint(0, item_num - 1)], img2_class)    
        # XXX: 注意should_get_same_class的值
        return img0_tuple[0], img1_tuple[0], img2_tuple[0]
    
    def classed_pack(self):
        # local = 'image/classed_pack/2019-03-14 16-30-img/'
        local = 'image/classed_pack/2019-03-14 16-30-color/'
        self.imageFolderDataset = []

        # floader1
        subFloader = os.listdir(local)
        for i in subFloader:
            temp = []
            sub_dir = local + i + '/'
            subsubFloader = os.listdir(sub_dir)
            for j in subsubFloader:
                temp.append(sub_dir + j)
            self.imageFolderDataset.append(temp)
        # floader2
        # 为数据集添加数据
        for i in range(self.__set_size__):
            img0, img1, img2 = self.__getitem__(len(self.imageFolderDataset))
            self.train_dataloader.append((img0, img1, img2))

if __name__ == '__main__':
    test_size=100
    print('Start preparing the data...')
    #colo初始化
    colo=Color()

    # 直方图
    hio=Histogram()

    #hash
    has=Hash()

    #CDH
    cdh=CDH()

    #MTH
    mth=MTH()
    #数据读入
    train_data = Dataset(set_size=test_size,batch_size=1)
    train_data.classed_pack()
    print('Finish preparing the data...')
    match_err = [0, 0, 0,0,0,0,]
    match_time = [0, 0, 0,0,0,0]
    for i, data in enumerate(train_data.train_dataloader):
        img0, img1, img2 = data
        print("Item:",i,end=" ")

        # start_time=time.time()
        # #CIEDE
        # dis_11=colo.color_test(img0, img1)
        # dis_10=colo.color_test(img0, img2)
        # if(dis_11>dis_10):
        #     match_err[1]=match_err[1]+1
        # match_time[1]=match_time[1]+(time.time()-start_time)

        # #Histogram
        start_time=time.time()
        dis_21=hio.color_test(img0, img1)
        dis_20=hio.color_test(img0, img2)
        if(dis_21<dis_20):
            match_err[2]=match_err[2]+1
        match_time[2]=match_time[2]+(time.time()-start_time)

        # #Hash
        # start_time=time.time()
        # dis_31=has.color_test(img0, img1)
        # dis_30=has.color_test(img0, img2)
        # if(dis_31>dis_30):
        #     match_err[3]=match_err[3]+1
        # match_time[3]=match_time[3]+(time.time()-start_time)

        # start_time=time.time()
        # dis_41=cdh.color_test(img0, img1)
        # dis_40=cdh.color_test(img0, img2)
        # print("CDH",end=" ")
        # if(dis_41<dis_40):
        #     match_err[4]=match_err[4]+1
        #     print("误判")
        # else:
        #     print(0)
        # match_time[4]=match_time[4]+(time.time()-start_time)

        # print("MTH",end=" ")
        # start_time=time.time()
        # dis_51=mth.color_test(img0, img1)
        # dis_50=mth.color_test(img0, img2)
        # if(dis_51<dis_50):
        #     match_err[5]=match_err[5]+1
        #     print("误判")
        # else:
        #     print(0)
        # match_time[5]=match_time[5]+(time.time()-start_time)
    print('Color:{:.3f}'.format(match_err[1]/test_size),end=' ')
    print('Histogram:{:.3f}'.format(match_err[2]/test_size),end=' ')
    print('Hash:{:.3f}'.format(match_err[3]/test_size),end=' ')
    print('CDH:{:.3f}'.format(match_err[4]/test_size),end=' ')
    print('MTH:{:.3f}'.format(match_err[5]/test_size),end=' ')

    print()
    print('TColor:{:.3f}'.format(match_time[1]/test_size),end=' ')
    print('THistogram:{:.3f}'.format(match_time[2]/test_size),end=' ')
    print('THash:{:.3f}'.format(match_time[3]/test_size),end=' ')
    print('TCDH:{:.3f}'.format(match_time[4]/test_size),end=' ')
    print('TMTH:{:.3f}'.format(match_time[5]/test_size),end=' ')