import os
import jieba
import re

if __name__ == '__main__':
    root_path = os.path.abspath('../')
    data_path1 = os.path.join(root_path, 'data\\AutoMaster_TestSet.csv')
    data_path2 = os.path.join(root_path, 'data\\AutoMaster_TrainSet.csv')
    stopwords_path = os.path.join(root_path, 'data\\stop_words.txt')
    write_path = os.path.join(root_path, 'data\\corpus_w2v.txt')

    with open(data_path1, 'r', encoding='utf-8') as f:
        data1 = f.readlines()

    with open(data_path2, 'r', encoding='utf-8') as f:
        data2 = f.readlines()

    stop_words = [word.strip() for word in open(stopwords_path, 'r', encoding='utf-8').readlines()]
    with open(write_path, 'w', encoding='utf-8') as f:
        temp1 = []
        for line in data1:
            line = list(jieba.cut(str(line).strip()))
            for word in line:
                temp1.append(word)
        f.write(' '.join(temp1))

        temp2 = []
        for line in data2:
            line = list(jieba.cut(str(line).strip()))
            for word in line:
                temp2.append(word)
        f.write(' '.join(temp2))

