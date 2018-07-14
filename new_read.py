# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:00:06 2018

@author: Crazy
"""

import os
import sys
import numpy as np
import cv2

import json
import time
from keras.models import model_from_json

IMAGE_SIZE = 64

#按照指定图像大小调整尺寸
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    print(image)
    #获取图像尺寸
    h, w, _ = image.shape

    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)    
    
    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    #RGB颜色
    BLACK = [0, 0, 0]
    
    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #调整图像大小并返回
    return cv2.resize(constant, (height, width))

#读取训练数据
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path):    #如果是文件夹，继续递归调用
            read_path(full_path)
        else:   #文件
            if dir_item.endswith('.bmp'):
                image = cv2.imread(full_path)   
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                
                images.append(image)                
                labels.append(path_name)                                
                    
    return images,labels
    

#从指定路径读取训练数据
def load_dataset(path_name):
    images,labels = read_path(path_name)    
    
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    #我和闺女两个人共1200张图片，IMAGE_SIZE为64，故对我来说尺寸为1200 * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    print(images.shape)    
    
    #标注数据，'me'文件夹下都是我的脸部图像，全部指定为0，另外一个文件夹，全部指定为1
    file_object = open('content.txt')
    name = file_object.read( ).split('\n')  
    for i in name:
        if i=='':
            name.remove('')

    """k = []
    for mark in range(0, len(name) - 1):
        for label in labels:
            if label.endswith(name[mark]):
                k=np.append(k,mark)
    """
    mark = 0
    labels = np.array([mark if label.endswith(name[mark]) else 1 for label in labels])    
    return images, labels

if __name__ == '__main__':
        images, labels = load_dataset('data')