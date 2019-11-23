from gensim.models import Word2Vec
import os
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'data_process'))
from data import Vocab
import numpy as np


def get_embedding(vocab_path, word_model_path, hps):
    vocab = Vocab(vocab_path, hps['vocab_size'])
    embed_size = hps['embed_size']
    model = Word2Vec.load(word_model_path)

    # encoder embedding
    embedding_matrix = np.zeros((hps['vocab_size'], embed_size))
    for i in range(hps['vocab_size']):
        word = vocab.id2word[i]
        if word not in model.wv.vocab:
            embedding_vector = np.random.uniform(-0.025, 0.025, (embed_size))
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = model.wv[word]
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


if __name__ == '__main__':
    import os
    import argparse

    word_model_path = os.path.join(os.path.abspath('../'), 'data', 'w2v.model')
    vocab_path = os.path.join(os.path.abspath('../'), 'data', 'words_frequences.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_enc_len", default=500, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=50, help="Decoder input max sequence length", type=int)
    parser.add_argument("--vocab_size", default=50000, help="Vocabulary size", type=int)
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--mode", default='train', help="mode")
    parser.add_argument("--embed_size", default=100, help="embed_size", type=int)
    args = parser.parse_args()
    hps = vars(args)
    embedding_matrix = get_embedding(vocab_path, word_model_path, hps)
    print(embedding_matrix[0])
    print(embedding_matrix.shape)


