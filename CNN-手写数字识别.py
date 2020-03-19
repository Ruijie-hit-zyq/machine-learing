import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
    (x_train, y_train),(x_test, y_test)=mnist.load_data()#x_train(60000, 28, 28)x_test(10000, 28, 28)
    number=10000
    x_train=x_train[0:number]#(10000, 28, 28)
    y_train=y_train[0:number]#(10000,)
    x_train=x_train.reshape(number,28*28)#(10000, 784),没懂变成啥了
    x_test=x_test.reshape(x_test.shape[0],28*28)#(10000, 784)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')

    y_train=np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train=x_train
    x_test=x_test#?

    x_train=x_train/255
    x_test=x_test/255
    x_test=np.random.normal(x_test)
    return (x_train, y_train),(x_test, y_test)


(x_train, y_train),(x_test, y_test)=load_data()

model=Sequential()
model.add(Dense(input_dim=28*28,units=10000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=100,epochs=20)

result=model.evaluate(x_test,y_test)
result2=model.evaluate(x_train,y_train)
print ('\nTrain Acc',result2[1])
print ('\nTest Acc',result[1])
