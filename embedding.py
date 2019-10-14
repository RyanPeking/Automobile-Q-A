import pandas as pd
from gensim.models import Word2Vec
from setting import max_features, max_len, embed_size, max_length_inp, max_length_targ, train_path, model_path
import tensorflow as tf


def load_w2v_model(filepath):
    model = Word2Vec.load(filepath)


def tokenize(sent, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, num_words = max_features)
    tokenizer.fit_on_texts(sent)  # 这样保证encoder和decoder使用的是同一个字典，按道理讲，decoder输出少，不应该维护那么长的字典
    word_index = tokenizer.word_index
    # print(word_index)
    # print(len(word_index))#一共十多万个词语，太多了
    tensor = tokenizer.texts_to_sequences(sent)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=max_len)
    return tensor, tokenizer, word_index


def load_dataset(data):
    source = [str(m) for m in data['input'].values.tolist()]
    target = [str(m) for m in data['Report'].values.tolist()]
    # inp_lang = data['Report'].values.tolist()[:num_examples]
    input_tensor, tokenizer1, word_index1 = tokenize(source, max_length_inp)
    target_tensor, tokenizer2, word_index2 = tokenize(target, max_length_targ)
    return input_tensor, target_tensor, word_index1, word_index2, tokenizer1, tokenizer2


def get_embedding():
    train_data = pd.read_csv(train_path, encoding='utf-8')
    model = load_w2v_model(model_path)

