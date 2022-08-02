#넘파이 불러와서 모델링~
#넘파이 불러와서 모델링~
import numpy as np
from keras.preprocessing.image import ImageDataGenerator #이미지데이터를 수치화


#1. 데이터

#수치화한 데이터를 저장한다.
# np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0]) # train x값
# np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1]) # train y값
# np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0]) # test x값
# np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1]) # test y값

x_train = np.load('d:/study_data/_save/_npy/project_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/project_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/project_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/project_test_y.npy')
test = np.load('d:/study_data/_save/_npy/project_test.npy')

print(x_train)
print(x_train.shape) #(3100, 150, 150, 3)
print(y_train.shape) #(3100, 30)
print(x_test.shape)  #(900, 150, 150, 3)
print(y_test.shape)  #(900, 30)

#2. 모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


model = Sequential()
model.add(Conv2D(100, (3, 3), input_shape=x_train.shape[1:], padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(100, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(100, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())   
model.add(Dense(512, activation='relu'))   # 출력
model.add(Dropout(0.5))

model.add(Dense(30, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=300, batch_size=128, validation_split=0.2) 

# 4. 평가 
score = model.evaluate(x_test, y_test)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


print('loss : ', loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', accuracy[-1])
print('val_loss : ', val_loss[-1])

# 5. 평가이미지 넣는곳

#predict할 이미지를 선언해준다
img = "img42" #정우성이미지로 테스트

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
from sklearn.metrics import accuracy_score
#acc=accuracy_score(y_test,x_test)
#print('acc score :', acc)

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
print(tabulate(df1.head(3), headers='keys', tablefmt='psql', showindex=True))
print("="*100)