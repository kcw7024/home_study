from email import header
from grpc import AuthMetadataContext
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import matplotlib.pyplot as plt
import cv2
from sympy import re
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout


#작업할 이미지들이 있는 폴더 경로를 선언 해준다.
actor_face_dir = 'D:/project/test_actor/test_actor/test_actor'

#클래스별 카테고리를 지정해준다.
categories = ['cho_seungwoo', 'choi_minsik','chun_woohee','e_som','go_ahseong',
              'ha_jungwoo','hwang_jeongmin','jeon_doyeon','jeon_jihyun','jeon_jongseo',
              'jo_inseong','jung_woosung','kang_dongwon','kang_hyejeong','kim_hyesoo',
              'kim_taeri','kim_yunseok','lee_byunghun','lee_jehoon','lee_jungjae',
              'moon_sori','park_sodam','shin_hakyun','son_seokgu','son_yejin',
              'song_hyekyo','song_kangho','suzy','yoo_haejin','yoon_yeojeong']

classes = len(categories) #클래스 카테고리 갯수를 선언해준다. # 30

#이미지 크기를 지정해준다.
image_w = 150 
image_h = 150
pixels = image_w * image_h * 3

# 이미지 작업된 것 들을 리스트로 반환 받음
x = [] 
y = []


#eunumerate :: 순서가있는 자료형을 입력받아 인덱스값을 포함하는 enumerate 객채를 리턴해줌
for idx, cat in enumerate(categories):
    label = [0 for i in range(classes)]     #카테고리를 0값으로 모두 지정해줌
    #print(label)
    label[idx] = 1                          #각 카테고리별 인덱스 값을 정해줌
    #print(label)
    image_dir = actor_face_dir + "/" + cat  #이미지 폴더 안에 지정한 각 카테고리포함
    #print(image_dir)
    files = glob.glob(image_dir + "/*.jpg") #해당 디렉토리 안에 지정한 확장자파일들을 리스트로 받아옴 
    for i, f in enumerate(files):           #자료형을 순서대로 입력받아 인덱스값을 포함하는 객체를 리턴
        img = Image.open(f)                 #폴더안에 이미지 열어서 작업
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)              #numpy 배열로 변환
        x.append(data)
        y.append(label)
        if i % 10 == 0:
            (i,"\n", data)
    
                                 
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
xy = (x_train, x_test, y_train, y_test)


#print('>>> data 저장중 ...')
np.save("d:/project/final.npy", xy)
#print("ok,", len(y))