# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:43 2018

@author: Crazy
"""

import cv2
import sys
import json
import time
import numpy as np
from keras.models import model_from_json
import os

import random

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from new_read import load_dataset, resize_image, IMAGE_SIZE
import gc
from train import Model
from train import Dataset


def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)                
    
    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break                
    
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像            
        
        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #大于0则检测到人脸                                   
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect                        
                
                #将当前帧保存为图片
                img_name = '%s/%d.bmp'%(path_name, num)                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                    break
                
                #画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (50,50,233),2)                
        
        #超过指定最大保存数量结束程序
        if num > (catch_pic_num): break                
                       
        #显示图像
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 
def mkdir(path): 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        os.makedirs(path) 
 
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False
 

   
if __name__ == '__main__':
    # 定义要创建的目录
    
    content = []
    
    while True:
        name = input('plz input two names(q to quit):')
        if name == 'q':
            print("over")
            break
        content.append(name)
        mkpath="data//"+name+"//"
        fobj=open("content.txt",'w')  
        fobj.writelines(('%s%s'%(str(items),os.linesep) for items in content))#content是要保存的list。
        fobj.close()
        print(content)
        mkdir(mkpath) 
        CatchPICFromVideo("Video", 0, 200, mkpath)
    #read
    images, labels = load_dataset('data')
    #train
    dataset = Dataset('data')    
    dataset.load()
    
    model = Model()
    model.build_model(dataset)

    #测试训练函数的代码
    model.train(dataset)
    #print("197 ok")
    model.save_model(file_path = './model/me.face.model.h5')
    model = Model()
    model.load_model(file_path = './model/me.face.model.h5')
    model.evaluate(dataset)
    #predict
    file_object = open('content.txt')
    name = file_object.read( ).split('\n')  
    for i in name:
        if i=='':
            name.remove('')
    model = Model()
    model.load_model(file_path = './model/me.face.model.h5')    
              
    #框住人脸的矩形边框颜色       
    color = (0, 255, 0)
    
    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    
    #人脸识别分类器本地存储路径
    cascade_path = "haarcascade_frontalface_default.xml"    
    
    #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频
        
        #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   
                
                for i in range(0,len(name)-1):
                    if faceID == i:                                                        
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                        cv2.putText(frame,name[faceID], 
                                    (x + 30, y + 30),                      #坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                    1,                                     #字号
                                    (50,50,233),                           #颜色
                                    2)                                     #字的线宽

                            
        cv2.imshow("Predict", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
    
    
