import cv2
import math
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cmc
from PIL import Image
import time
import random
import os
import sys
sys.path.append('..')

from CDH import CDH
from MTH import MTH
from Hash import Hash
from Histogram import Histogram
from Hist_CIEDE import Color

change_scale=1

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
        local = 'test/img/'
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
    # 测试次数
    test_size=1
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

        start_time=time.time()
        #CIEDE
        dis_11=colo.color_test(img0, img1)
        dis_10=colo.color_test(img0, img2)
        if(dis_11>dis_10):
            match_err[1]=match_err[1]+1
        match_time[1]=match_time[1]+(time.time()-start_time)

        # #Histogram
        start_time=time.time()
        dis_21=hio.color_test(img0, img1)
        dis_20=hio.color_test(img0, img2)
        if(dis_21<dis_20):
            match_err[2]=match_err[2]+1
        match_time[2]=match_time[2]+(time.time()-start_time)

        #Hash
        start_time=time.time()
        dis_31=has.color_test(img0, img1)
        dis_30=has.color_test(img0, img2)
        if(dis_31>dis_30):
            match_err[3]=match_err[3]+1
        match_time[3]=match_time[3]+(time.time()-start_time)

        start_time=time.time()
        dis_41=cdh.color_test(img0, img1)
        dis_40=cdh.color_test(img0, img2)
        if(dis_41<dis_40):
            match_err[4]=match_err[4]+1
            print("误判")
        else:
            print(0)
        match_time[4]=match_time[4]+(time.time()-start_time)

        start_time=time.time()
        dis_51=mth.color_test(img0, img1)
        dis_50=mth.color_test(img0, img2)
        if(dis_51<dis_50):
            match_err[5]=match_err[5]+1
            print("误判")
        else:
            print(0)
        match_time[5]=match_time[5]+(time.time()-start_time)
    print('错误率')
    print('Color:{:.3f}'.format(match_err[1]/test_size),end=' ')
    print('Histogram:{:.3f}'.format(match_err[2]/test_size),end=' ')
    print('Hash:{:.3f}'.format(match_err[3]/test_size),end=' ')
    print('CDH:{:.3f}'.format(match_err[4]/test_size),end=' ')
    print('MTH:{:.3f}'.format(match_err[5]/test_size),end=' ')

    print('耗时')
    print('TColor:{:.3f}s'.format(match_time[1]/test_size),end=' ')
    print('THistogram:{:.3f}s'.format(match_time[2]/test_size),end=' ')
    print('THash:{:.3f}s'.format(match_time[3]/test_size),end=' ')
    print('TCDH:{:.3f}s'.format(match_time[4]/test_size),end=' ')
    print('TMTH:{:.3f}s'.format(match_time[5]/test_size),end=' ')