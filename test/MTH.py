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
