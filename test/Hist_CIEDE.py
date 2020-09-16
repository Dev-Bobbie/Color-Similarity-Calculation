import cv2
import math
from math import sqrt
import numpy as np
from colormath.color_objects import LabColor
from operator import itemgetter
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cmc
from PIL import Image
import time
import random
import os
import sys
sys.path.append('..')
change_scale=1
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
