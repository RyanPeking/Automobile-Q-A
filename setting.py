import os

max_features = 300
max_len = 300
embed_size = 100
max_length_inp,max_length_targ= 500, 50
train_path = os.path.join(os.path.abspath('./'), 'data', 'train.csv')
model_path = os.path.join(os.path.abspath('./'), 'data', 'w2v.model')