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

#배치를 최대로 잡으면 이방법도 가능

hist = model.fit(x_train, y_train, epochs=300, batch_size=128, validation_split=0.2) 

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


print('loss : ', loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', accuracy[-1])
print('val_loss : ', val_loss[-1])


#4. 평가, 훈련


y_predict = model.predict(test)
print(y_predict)
y_predict = np.argmax(y_predict,axis=1)
print(y_predict)