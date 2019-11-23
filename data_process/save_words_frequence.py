import os
from collections import Counter
import pandas

def save_word_dict(words_frequence, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word, frequence in words_frequence:
            f.write("%s\t%d\n" % (word, frequence))


def read_data(train_path, test_path):
    train = pandas.read_csv(train_path)
    test = pandas.read_csv(test_path)

    words = []
    for line in train['input']:
        words += line.split()

    for line in train['Report']:
        words += line.split()

    for line in test['input']:
        words += line.split()

    return words


def build_vocab(words):
    words_frequence = Counter(words)
    words_frequence = sorted(words_frequence.items(), key=lambda d: d[1], reverse=True)
    return words_frequence


if __name__ == '__main__':
    root_path = os.path.abspath('../')
    words = read_data(os.path.join(root_path, 'data', 'treated_train.csv'), os.path.join(root_path, 'data', 'test.csv'))
    words_frequence = build_vocab(words)
    save_word_dict(words_frequence, os.path.join(root_path, 'data', 'words_frequences.txt'))
