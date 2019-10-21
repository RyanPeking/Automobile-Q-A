# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import os
import pickle
from typing import Tuple, Callable, Dict, Union
import time

class Encoder(tf.keras.Model):
    def __init__(self,vocab_size,
                 vec_dim,
                 matrix,
                 gru_size):
        super(Encoder,self).__init__()
        # embedding_weights = None
        weights = [matrix]
        self.embedding = tf.keras.layers.Embedding(vocab_size, vec_dim, weights=weights, trainable=False)


        self.gru = tf.keras.layers.GRU(gru_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 初始化https://zhuanlan.zhihu.com/p/25110150
        # google: Xavier Glorot


        self.gru_size = gru_size
    def call(self,sequence,states):
        embed = self.embedding(sequence)
        # output,state_h,context_v = self.gru(embed,initial_state=states)
        output, state_h = self.gru(embed, initial_state=states)
        # states:hidden
        return output,state_h
    def init_states(self,batch_size:int):
        return tf.zeros([batch_size, self.gru_size])


class BahdanauAttention(tf.keras.Model):
    # other attention is LuongAttention
    def __init__(self, units):
        # unit: gru_size
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: state_h
        # values：output
        # hidden_shape == (batch_size, gru_size) query   units == hidden_size
        # value == (batch_size, max_length, gru_size)    max_length == sequence_length
        # hidden_with_time_axis_shape == (batch_size, 1, gru_size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score_shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, gru_size)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weight = tf.nn.softmax(score, axis=1)

        context_vector = attention_weight * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)

        return context_vector, attention_weight

class Decoder(tf.keras.Model):
    def __init__(self,vocab_size,
                 vec_dim,
                 matrix,
                 gru_size):
        super(Decoder,self).__init__()
        self.gru_size = gru_size
        weights = [matrix]
        self.embedding = tf.keras.layers.Embedding(vocab_size, vec_dim,
                                                   # embeddings_initializer=tf.keras.initializers.Constant(
                                                   #     weights),
                                                   weights=weights,
                                                   trainable=False)
        self.attention = BahdanauAttention(self.gru_size)
        self.gru = tf.keras.layers.GRU(self.gru_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.wc = tf.keras.layers.Dense(self.gru_size,activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self,sequence,state,encoder_output):
        embed = self.embedding(sequence)
        # gru_out, state_h = self.gru(embed, initial_state=state)
        gru_out, state_h = self.gru(embed)
        context_vector, attention_weight = self.attention(gru_out,encoder_output)

        gru_out = tf.concat([tf.squeeze(context_vector, 1), tf.squeeze(gru_out, 1)], 1)
        # 删除所有大小是1的维度
        # (batch_size, embedding_dim + hidden_size)

        '''
        tf.squeeze
        't' is a tensor of shape [1, 2, 1, 3, 1, 1]
        shape(squeeze(t)) ==> [2, 3]
        Or, to remove specific size 1 dimensions:

        't' is a tensor of shape [1, 2, 1, 3, 1, 1]
        shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
        '''

        gru_out = self.wc(gru_out)
        # output shape == (batch_size, gru_size)
        # attention_vector

        logits = self.ws(gru_out)
        # output shape == (batch_size, vocab)

        return logits, state_h, attention_weight

    # def init_states(self,batch_size):
    #     return (tf.zeros([batch_size,self.gru_size]),
    #             tf.zeros([batch_size,self.gru_size]))

from embedding import get_embedding


embedding_matrix1,embedding_matrix2,input_tensor,target_tensor,tokenizer1,tokenizer2 = get_embedding()
def data_loader(input_tensor,target_tensor):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
    dataset = dataset.batch(len(target_tensor), drop_remainder=True)
    # example_input_batch, example_target_batch = next(iter(dataset))
    # example_input_batch.shape, example_target_batch.shape
    return dataset



class Auto_model:
    def __init__(self,input_tensor,
                 target_tensor,
                 batch_size,
                 embedding_matrix1,
                 embedding_matrix2,
                 tokenizer1,
                 tokenizer2,
                 unit):
        self.BUFFER_SIZE = len(input_tensor)
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.encoder_embedding = embedding_matrix1
        self.decoder_embedding = embedding_matrix2
        self.batch_size = batch_size
        self.steps_per_epoch = len(input_tensor) // batch_size
        self.embedding_dim = embedding_matrix1.shape[1]
        self.unit = unit
        self.vocab_inp_size = embedding_matrix1.shape[0]
        self.vocab_tar_size = embedding_matrix2.shape[0]
        self.tokenizer_encoder = tokenizer1
        self.tokenizer_decoder = tokenizer2

        example_input_batch, example_target_batch = self.get_batch()
        self.build_network()

    def get_batch(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor, self.target_tensor)).\
            shuffle(len(self.input_tensor))
        self.dataset = dataset.batch(len(self.target_tensor), drop_remainder=True)
        example_input_batch, example_target_batch = next(iter(self.dataset))
        return example_input_batch, example_target_batch

    def build_network(self):
        #encoder part
        example_input_batch, example_target_batch = self.get_batch()
        self.encoder = Encoder(self.vocab_inp_size,self.embedding_dim,self.encoder_embedding,self.unit)
        sample_hidden = self.encoder.init_states(self.batch_size)
        output, state_h = self.encoder(example_input_batch, sample_hidden)

        #attention part
        self.attention_layer = BahdanauAttention(2)
        attention_result, attention_weights = self.attention_layer(state_h, output)

        #decoder part
        self.decoder = Decoder(self.tokenizer_decoder,self.embedding_dim,self.decoder_embedding,self.unit)
        logits, state_h, state_c, aligment = self.decoder(tf.random.uniform((self.batch_size, 1)), sample_hidden,
                                                           output)


    @tf.function
    def train_loss_op(self,inp, targ, enc_hidden):
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        checkpoint_dir = './model_save'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            # 不是0标记为True, 0标记为False
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden= self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.tokenizer_decoder.word_index['<start>']] * self.batch_size, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def run_op(self,epochs):
        for epoch in epochs:
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_loss_op(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    model = Auto_model(input_tensor,
                 target_tensor,
                 16,
                 embedding_matrix1,
                 embedding_matrix2,
                 tokenizer1,
                 tokenizer2,
                 8)

    model.run_op(1)








