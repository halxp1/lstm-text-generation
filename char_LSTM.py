import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from leras.models import load_model


raw_text = open('input/Winston_Churchil.txt',encoding='utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i , c in enumerate(chars))
int_to_char = dict((i,c) for i, c in enumerate(chars))

print(chars)
print(len(raw_text))

seq_length = 100
x=[]
y=[]

for i in range(0, len(raw_text)-seq_length):
    given = raw_text[i:i+seq_length]
    predict = raw_text[i+seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

print(x[:3])
print(y[:3])

n_patterns = len(x)
n_vocab = len(chars)

# 把x变成LSTM需要的样子
x= np.array(x)
x =np.reshape(x,(n_patterns,seq_length,1))
x = x/float(n_vocab)

y=np_utils.to_categorical(y)

print(x[11])
print(y[11])

#模型构建
#LSTM模型构建
model =Sequential()
model.add(LSTM(128,input_shape=(x.shape[1],x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#跑模型
model.fit(x,y,epochs=10,batch_size=32)
model.save('my_model.h5')

#我们来写个程序,看看我们训练出来的LSTM的效果：
def predict_next(input_array):
    input_array=np.array(list(input_array))
    x=np.reshape(input_array, (1,seq_length,1))
    x = x/float(n_vocab)
    y=model.predict(x)
    return y

def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input)-seq_length):]:
        res.append(char_to_int[c])
    return res

def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c

def generate_article(init, rounds=500):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string +=n
    return in_string

init = 'Professor Michael S. Hart is the originator of the Project'
article = generate_article(init)

print(article)





















