import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import json
import matplotlib.pyplot as pyplot
from PIL import Image,ImageDraw,ImageFont
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import csv
#import keras
#from keras.models import load_model
# 指定使用字型和大小
myfont = FontProperties(fname='C:/Windows/Fonts/mingliu.ttc', size=40)

#讀取openpose的套件
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


def show_Wrist(x_count,x_test,y_test):
    # 設定圖片大小為長15、寬10
    plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
    # 把資料放進來並指定對應的X軸、Y軸的資料，用方形做標記(s-)，並指定線條顏色為紅色，使用label標記線條含意
    plt.plot(x_count,x_test,'s-',color = 'r', label="4-x")
    plt.plot(x_count,y_test,'s-',color = 'g', label="0-x")
    # 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
    plt.title("反手-手腕身體變化圖", fontproperties=myfont, x=0.5, y=1.03)
    # 设置刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 標示x軸(labelpad代表與圖片的距離)
    plt.xlabel("count", fontsize=30, labelpad = 15)
    # 標示y軸(labelpad代表與圖片的距離)
    plt.ylabel("數值",fontproperties=myfont, fontsize=30, labelpad = 20)
    
    # 顯示出線條標記位置
    plt.legend(loc = "best", fontsize=20)
    # 畫出圖片
    plt.show()

def check_path_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def acc(a):
    a = np.array(a)
    x,y=a.shape
    for i in range(0,x-1):
        for j in range(0,y):
            a[i][j]=a[i+1][j]-a[i][j]
    a=np.delete(a, -1, axis=0)
    
    return a
def acc_1d(a):
    x=len(a)
    for i in range(0,x-1):
        a[i]=a[i+1]-a[i]
    a.pop()
    
    return a

def res(arr):
    zeroarray=np.zeros((1,34))
    x,y=arr.shape
    #print(arr.shape)
    k=120-x+1
    if x<=120:
        for i in range(1,k):
            arr = np.vstack((arr,zeroarray))      #重設矩陣大小
    return arr

# 資料生成

path = 'D:/Tennis/F2'                        #讀取該目錄中的mp4檔案
for root, dirs, names in os.walk(path):
    for name in names:
        ext = os.path.splitext(name)[1]
        if ext == '.mp4':
            fromdir = os.path.join(root, name)                                      #D:/openpose-master/examples/media/test/ 中的mp4檔案
            openpose=path+"openpose/"                                               #D:/openpose-master/examples/media/test/openpose
            openpose_json=openpose+os.path.basename(fromdir).replace('.mp4', '')    #D:/openpose-master/examples/media/test/openpose/ + aaa.mp4(替換.mp4)
            npy_path=path+"/output_npy_new_or/"                                               #D:/openpose-master/examples/media/test/output
            video_path=path+"/video/"                                                #D:/openpose-master/examples/media/test/video
            mp4_name = os.path.basename(fromdir).replace('.mp4', '')                #aaa
            print(fromdir)
            #print(openpose_json)
            #cv2.waitKey(3000)
            #D:/openpose-master/examples/media/test/openpose/ + aaa.mp4(替換.mp4)
            #if not os.path.isdir(openpose_json):
            #    os.makedirs(openpose_json)

            check_path_exist(video_path)
            check_path_exist(npy_path)
            
            params = dict()                                             #設定openpose內部的一些參數
            params["model_folder"] = "../../../models/"
            params["display"] = 0
            params["model_pose"] = "BODY_25"
            params["number_people_max"] = 1                             #偵測到的最大人數

            #params["write_json"] = openpose_json                       #影片keypoints的json檔

            try:
                opWrapper = op.WrapperPython()                          #建立OpenPose
                opWrapper.configure(params)                             #將opWrapper設定剛剛給的參數
                opWrapper.start()                                       #開始openpose

                datum = op.Datum()
                cap = cv2.VideoCapture(fromdir)                         #將讀取到的路徑使用VideoCapture讀取
                fps = cap.get(cv2.CAP_PROP_FPS)                         #獲取該影片的fps
                video=None
                count=0 
                fps_time=0
                max_count=180
                x=0
                y=0
                
                x_count=[]
                x_test=[]
                y_test=[]

                register=np.empty((1,34))
                while (cap.isOpened()):
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      #影片的寬
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    #影片的高
                    total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1#影片的總偵數
                    hasframe, frame= cap.read()
                    cnn_count=0
                    
                    
                    if hasframe== True:

                        datum.cvInputData = frame                       #將讀取到的影片給openpose執行
                        datum.name=str(count) 
                        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                        print("Body keypoints: \n" + str(datum.poseKeypoints))
                        
                        a=datum.poseKeypoints[0]                        #將三維轉二維
                        #print(a)
                        dataset = np.delete(a, -1, axis=1)              #移除誤差值
                        if dataset[4][0] >0:
                            count=count+1
                            #origin=np.delete(dataset, [15,16,17,18,20,21,23,24], axis=0)                          #將不必要的關鍵點移除
                            x_count.append(count)
                            x_test.append(dataset[4][0])
                            y_test.append(dataset[1][0])
                            print("Body keypoints:  \n",count)              #印出keypoints
                            dataset.resize(1,50)
                            p=np.round(dataset,decimals=3)
                            
                            print(p)
                            print(p.shape)
                            #print("\n")
                            
                            
                            if(count==1):
                                register = dataset
                            if(count>1):
                                register = np.vstack((register,dataset))
                            #rigster=origin

                            print(register)
                            print(register.shape)

                        else:
                            total_count-=1
                        
                        key = cv2.waitKey(1) & 0xFF
                        opframe=datum.cvOutputData
    
                        cv2.putText(opframe,
                                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                                    (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)

                        cv2.putText(opframe,
                                    "count: %d" % count,
                                    (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)
                        
                        cv2.imshow("OpenPose 1.5.0 ", opframe)          #展示影片
                        fps_time = time.time()
                        cv2.waitKey(25)
                        
                        

                        video_path=video_path+mp4_name+".avi"

                        if video == None:                               #將影片儲存
                            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        video.write(opframe)

                    else:
                        break
                print(register)
                
                #register=acc(register)                                  #將數值轉為速度
                #print(register)
                #print(register.shape)

                #register=acc(register)                                  #將數值轉為加速度
                #print(register)
                #print(register.shape)


                #register=res(register)                                  #重設矩陣大小
                #print(register)
                #print(register.shape)





                cv2.destroyAllWindows()
                video.release()
                
                #acc_1d(x_test)
                #acc_1d(x_test)
                #acc_1d(y_test)
                #acc_1d(y_test)
                #x_count.pop()
                #x_count.pop()
                #印出手腕變化圖
                #show_Wrist(x_count,x_test,y_test)

                    
                #print(rigster)
                
                #register=register.flatten('F')
                #print(register)
                
                #print(register.size)
                
                #將輸出關節點儲存成npy檔
                npy_name=npy_path+mp4_name+".npy"
                np.save(npy_name, register)

                f=np.load(npy_name)
                print(f)
                

                #print(test)
                #test_f=test.flatten('F')                          #將二維轉一維
                #print(test_f)
                #print(test_f.shape)
                
            except Exception as e:
                print(e)
                sys.exit(-1)
print("END")
