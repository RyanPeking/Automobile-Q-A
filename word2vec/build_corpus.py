import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train_w2v_model(txt_path, model_path):
    w2v_model = Word2Vec(LineSentence(txt_path), workers=4, min_count=5)
    w2v_model.save(model_path)


def get_model_from_file(model_path):
    model = Word2Vec.load(model_path)
    return model

if __name__ == '__main__':
    root_path = os.path.abspath('../')
    txt_path = os.path.join(root_path, 'data\\corpus_w2v.txt')
    model_path = os.path.join(root_path, 'data\\w2v.model')

    train_w2v_model(txt_path, model_path)