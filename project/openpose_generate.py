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
import keras
from keras.models import load_model
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

def check_path_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

#將要測試的資料放進某個資料夾，並將其路徑貼在path變數中後在執行程式
#    |
#    |
#  \ | /
#   \|/ 
path = 'D:/Tennis/test-3'                        #讀取該目錄中的mp4檔案
for root, dirs, names in os.walk(path):
    for name in names:
        ext = os.path.splitext(name)[1]
        if ext == '.mp4':
            fromdir = os.path.join(root, name)                                      #D:/openpose-master/examples/media/test/ 中的mp4檔案
            openpose=path+"openpose/"                                               #D:/openpose-master/examples/media/test/openpose
            openpose_json=openpose+os.path.basename(fromdir).replace('.mp4', '')    #D:/openpose-master/examples/media/test/openpose/ + aaa.mp4(替換.mp4)
            npy_path=path+"/output_npy/"                                               #D:/openpose-master/examples/media/test/output
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
                

                register=np.empty((2,25))
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

                        #print("Body keypoints: \n" + str(datum.poseKeypoints))
                        
                        a=datum.poseKeypoints[0]                        #將三維轉二維
                        
                        dataset = np.delete(a, -1, axis=1)              #移除誤差值
                        if dataset[4][0] >0:
                            count=count+1
                            x=dataset.flatten('F')                          #將二維轉一維

                            #print("Body keypoints:  \n",count)              #印出keypoints
                            
                            p=np.round(x,decimals=3)
                            #print(p)
                            #print("\n")
                            
                            
                            if(count==1):
                                register = x
                            if(count>1):
                                register = np.vstack((register,x))
                            #rigster=origin

                            #print(register)

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
                        '''
                        if count>10:
                            cv2.putText(opframe,
                                    "test",
                                    (200, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)
                        '''
                        
                        
                        if count == total_count-1:
                            max=120
                            data=[]
                            cnn_data=register.flatten('F')
                            len=(cnn_data.size)/50
                            if len<max :
                                cnn_data.resize(120*50)
                                cnn_data=cnn_data.reshape((120*50,1))
                            data.append(cnn_data)
                            data = np.array(data) # 3
                            print(data.shape)
                            modello = load_model('1d_cnn.h5')

                            Y_pred = modello.predict(data)
                            Y_pred = np.argmax(Y_pred,axis=1)
                            print(Y_pred)
                            if Y_pred == 0:
                                cnn_count=1
                            if Y_pred == 1:
                                cnn_count=2
                        
                        if count>=total_count-3:
                            #print(cnn_count)
                            if cnn_count == 1:
                                print("forehand")
                                cv2.putText(opframe,
                                    "forehand",
                                    (300, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)
                            if cnn_count == 2:
                                print("backhand")
                                cv2.putText(opframe,
                                    "backhand",
                                    (300, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
                
                
                cv2.destroyAllWindows()
                video.release()
                
                #print(rigster)
                
                #register=register.flatten('F')
                #print(register)
                
                #print(register.size)
                
                #將輸出關節點儲存成csv檔
                #npy_name=npy_path+mp4_name+".npy"
                #np.save(npy_name, register)


                #f=np.load(npy_name)
                #print(f)
                '''
                #矩陣處理
                a=x-1
                b=y
                test=np.empty((a,b))
                i=1
                j=1
                for i in range(a):
                    for j in range(b):
                        if(i>x):
                            break
                        test[i][j]=rigster[i+1][j]-rigster[i][j]
                '''

                #print(test)
                #test_f=test.flatten('F')                          #將二維轉一維
                #print(test_f)
                #print(test_f.shape)
                
                '''
                #將矩陣轉成圖片
                img = np.zeros([a, b, 3], dtype=np.uint8)
                for i in range(a):
                        for j in range(b):
                            img[i, j, :] = [test[i][j],test[i][j],test[i][j]]        

                cv2.imshow('test',img)
                print(img.shape)
                cv2.waitKey(25)
                cv2.destroyAllWindows()
                img_name=img_path+mp4_name+".jpg"
                #儲存圖片
                cv2.imwrite(img_name, img)
                print(img_name)

                '''
            except Exception as e:
                print(e)
                sys.exit(-1)
print("END")
