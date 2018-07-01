# lstm-text-generation
文本生成(Word2Vec + RNN/LSTM)

# 目录：
      input : 输入文件数据
      1.char_LSTM.py  : 以字母为维度 预测下一个字母是什么
      2.word_LSTM.py  : 以单词为维度，预测下一个单词是是什么

# char_LSTM.py
  用RNN做文本生成，我们这里用温斯顿丘吉尔的任务传记作为我们的学习语料。
  英文的小说语料可以从古登堡计划网站下载txt平文本：https://www.gutenberg.org/wiki/Category:Bookshelf)
  这里我们采用keras简单的搭建深度学习模型进行学习。