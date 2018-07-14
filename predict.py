# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:42:39 2018

@author: Crazy
"""

#-*- coding: utf-8 -*-

import cv2
import sys
import gc
from train import Model
import numpy

if __name__ == '__main__':      
    #加载模型
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