import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

#接下来，我们把文本读入
raw_text = ''
for file in os.listdir("input"):
	if file.endswith(".txt"):
		raw_text+=open("input/"+file,errors='ignore',encoding='utf-8').read()+'\n\n'   #读入多个文件

raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
sents = sentensor.tokenize(raw_text)  #fen
corpus = []
for sen in sents:
	corpus.append(nltk.word_tokenize(sen))

print(len(corpus))
print(corpus[:3])

w2v_model = Word2Vec(corpus,size=128, window = 5, min_count = 5, workers = 4)

print(w2v_model['office'])

# 把源数据变成一个长长的x，好让LSTM学会predict下一个单词

raw_input = [item for sublist in corpus for item in sublist]

print(raw_input[:100])

text_stream = []
vocab = w2v_model.wv.vocab
for word in raw_input:
	if word in vocab:
		text_stream.append(word)
len(text_stream)

#这里我们的文本预测积就是，给出了前面的单次以后，下一个单词是谁？
#比我，hello from the other ,给出side

#构造训练测试集
#x 是前置字母们 y 是后一个字母

seq_length=10
x=[]
y=[]
for i in range(0, len(text_stream)-seq_length):
	given = text_stream[i:i+seq_length]
	predict = text_stream[i+seq_length]
	x.append(np.array([w2v_model[word] for word in given]))
	y.append(w2v_model[predict])

print(x[10])
print(y[10])

print(len(x))
print(len(y))

print(len(x[12]))
print(len(x[12][0]))
print(len(y[12]))
x=np.array(x)
y=np.array(y)
x = np.reshape(x,(-1,seq_length,128))
y = np.reshape(y,(-1,128))

# 接下来我们做两件事：
#
#    1我们已经有了一个input的数字表达（w2v），我们要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征]
#
#    2第二，对于output，我们直接用128维的输出

model = Sequential()
model.add(LSTM(256,dropout_W=0.2, dropout_U=0.2,input_shape=(seq_length,128)))
model.add(Dropout(0.2))
model.add(Dense(128,activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')

#跑模型
model.fit(x,y,epochs=50,batch_size=4096)

#接下来写个程序，看看我们训练出来的LSTM的效果：

def predict_next(input_array):
	input_array=np.array(input_array)
	x = np.reshape(input_array,(-1,seq_length,128))
	y = model.predict(x)
	return y

def string_to_index(raw_input):
	raw_input = raw_input.lower()
	input_stream = nltk.word_tokenize(raw_input)
	res = []
	for word in input_stream[(len(input_stream)-seq_length):]:
		res.append(w2v_model[word])
	return res

def y_to_word(y):
	word = w2v_model.most_similar(positive = y,topn=1)
	return word

#写一个大程序
def generate_article(init,rounds=30):
	in_string = init.lower()
	for i in range(rounds):
		n=y_to_word(predict_next(string_to_index(in_string)))
		in_string +=' '+n[0][0]
	return in_string

init = 'Language Models allow us to measure how likely a sentence is, which is an important for Machine'
article = generate_article(init)
print(article)




























