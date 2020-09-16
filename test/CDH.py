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
