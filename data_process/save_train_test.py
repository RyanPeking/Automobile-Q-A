import os
import jieba
import pandas as pd


def seg_line(line, add_stopwords=False):
    if add_stopwords:
        stopwords_path = os.path.join(root_path, 'data', 'stop_words.txt')
        stop_words = [word.strip() for word in open(stopwords_path, 'r', encoding='utf-8').readlines()]
        tokens = jieba.cut(str(line))
        words = [word for word in tokens if word not in stop_words]
        words = [word for word in tokens if word]
        return ' '.join(words)
    else:
        tokens = jieba.cut(str(line))
        words = [word for word in tokens if word]
        return ' '.join(words)


def split_data(train_path, test_path, train_save_path, test_save_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.dropna(axis=0, how='any', inplace=True)
    test.dropna(axis=0, how='any', inplace=True)

    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    for k in ['Brand', 'Model', 'Question', 'Dialogue', 'Report']:
        for i in range(len(train[k])):
            line = train[k].get(i)
            # line = re.findall(r'[\w]+', str(line))
            # line = ''.join(line)
            train[k][i] = seg_line(line)

    for k in ['Brand', 'Model', 'Question', 'Dialogue']:
        for i in range(len(test[k])):
            line = test[k].get(i)
            # line = re.findall(r'[\w]+', str(line))
            # line = ''.join(line)
            test[k][i] = seg_line(line)

    train.dropna(axis=0, how='any', inplace=True)
    test.dropna(axis=0, how='any', inplace=True)

    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    train['input'] = train['Brand'] + ' ' + train['Model'] + ' ' + train['Question'] + ' ' + train['Dialogue']
    test['input'] = test['Brand'] + ' ' + test['Model'] + ' ' + test['Question'] + ' ' + test['Dialogue']

    train.drop(['Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)
    test.drop(['Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)

    train.to_csv(train_save_path, index=False, encoding='utf-8')
    test.to_csv(test_save_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    root_path = os.path.abspath('../')
    data_path1 = os.path.join(root_path, 'data', 'AutoMaster_TrainSet.csv')
    data_path2 = os.path.join(root_path, 'data', 'AutoMaster_TestSet.csv')

    train_save_path = os.path.join(root_path, 'data', 'treated_train.csv')
    test_save_path = os.path.join(root_path, 'data', 'test.csv')

    split_data(data_path1, data_path2, train_save_path, test_save_path)