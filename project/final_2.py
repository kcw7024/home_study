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


actor_face_dir = 'D:/project/test_actor/test_actor/test_actor'

#클래스별 카테고리를 지정해준다.
categories = ['cho_seungwoo', 'choi_minsik','chun_woohee','e_som','go_ahseong',
              'ha_jungwoo','hwang_jeongmin','jeon_doyeon','jeon_jihyun','jeon_jongseo',
              'jo_inseong','jung_woosung','kang_dongwon','kang_hyejeong','kim_hyesoo',
              'kim_taeri','kim_yunseok','lee_byunghun','lee_jehoon','lee_jungjae',
              'moon_sori','park_sodam','shin_hakyun','son_seokgu','son_yejin',
              'song_hyekyo','song_kangho','suzy','yoo_haejin','yoon_yeojeong']

classes = len(categories) #클래스 카테고리 갯수를 선언해준다. # 30


image_w = 64
image_h = 64
# 데이터 열기 
x_train, x_test, y_train, y_test = np.load("d:/project/final.npy", allow_pickle=True) #npy파일 불러올때 에러가 난다. allow_pickle 을 True로 해주면 해결됨
# 데이터 정규화하기(0~1사이로)
x_train = x_train.astype("float") / 256
x_test = x_test.astype("float") / 256
#print(x_train.shape) #(2250, 150, 150, 3) 
#print(x_train.shape[1:]) 

#2. 모델
 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())   
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

print(model.summary())


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, 
                 epochs=200, 
                 validation_split=0.2
                 )

# 4. 평가 
score = model.evaluate(x_test, y_test)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 5. 평가이미지 넣는곳

#predict할 이미지를 선언해준다
img = "3 (39)" 

test_image = 'd:/project/img/'+img+'.jpg'

#평가할 이미지 수치화
img = Image.open(test_image)
img = img.convert("RGB")
img = img.resize((150,150))
data = np.asarray(img)
x = np.array(data)
x = x.astype("float") / 256
x = x.reshape(-1, 150, 150,3)

pred = model.predict(x)  
#print(pred)
result = [np.argmax(value) for value in pred]   #예측값중 가장높은 클래스 반환
#print(result)

print("="*100)
print('loss : ', loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', accuracy[-1])
print('val_loss : ', val_loss[-1])
print("="*100)
print('배우이름 : ',categories[result[0]])


#영화추천
import pandas as pd
from tabulate import tabulate
path = 'C:/home_study/home_study/'
recommend  = pd.read_csv(path + 'actor_filmo_n_score.csv')

df1 = pd.DataFrame(recommend) #불러온 csv 파일 데이터프레임으로 변경
df1 = df1.sort_values(by='평점', ascending=False) #평점높은순으로 정렬함
df1 = df1.drop(columns=['Unnamed: 0']) #필요없는 컬럼은 지워줌
print("="*100)
#평가된 결과값과 동일한 배우코드의 필모그래피중 평점이 5.0이상인 영화 1개를 출력
df1 = df1[(df1['배우코드'] == categories[result[0]]) & (df1['평점']>5.0)]
#직관적으로 보이게 하기 위해 tabulate 를 사용
print(tabulate(df1.head(1), headers='keys', tablefmt='psql', showindex=True))
print("="*100)