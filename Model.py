import os
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
        '''
        keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None,
                               activity_regularizer=None, embeddings_constraint=None, mask_zero=False,
                               input_length=None)
        
        参数
        input_dim: int > 0。词汇表大小， 即，最大整数 index + 1。
        output_dim: int >= 0。词向量的维度。
        embeddings_initializer: embeddings 矩阵的初始化方法 (详见 initializers)。   https://keras.io/zh/initializers/
        embeddings_regularizer: embeddings matrix 的正则化方法 (详见 regularizer)。
        embeddings_constraint: embeddings matrix 的约束函数 (详见 constraints)。
        mask_zero: 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。 这对于可变长的 循环神经网络层 十分有用。 
        如果设定为 True，那么接下来的所有层都必须支持 masking，否则就会抛出异常。 如果 mask_zero 为 True，作为结果
        ，索引 0 就不能被用于词汇表中 （input_dim 应该与 vocabulary + 1 大小相同）。
        input_length: 输入序列的长度，当它是固定的时。 如果你需要连接 Flatten 和 Dense 层，则这个参数是必须的 
        （没有它，dense 层的输出尺寸就无法计算）。
        '''

        self.gru = tf.keras.layers.GRU(gru_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        '''
        return_sequences：默认 False。在输出序列中，返回单个 hidden state值还是返回全部time step 的 hidden state值。
         False 返回单个， true 返回全部。
        return_state：默认 False。是否返回除输出之外的最后一个状态。
        
        lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)
        此时，我们既要输出全部时间步的 hidden state ，又要输出 cell state。        
        lstm1 存放的就是全部时间步的 hidden state。        
        state_h 存放的是最后一个时间步的 hidden state        
        state_c 存放的是最后一个时间步的 cell state

        '''


        '''
        keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', 
        kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, 
        implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, 
        reset_after=False)
        units: 正整数，输出空间的维度。
        activation: 要使用的激活函数 (详见 activations)。 默认：双曲正切 (tanh)。 如果传入 None，
        则不使用激活函数 (即 线性激活：a(x) = x)。
        recurrent_activation: 用于循环时间步的激活函数 (详见 activations)。 
        默认：分段线性近似 sigmoid (hard_sigmoid)。 如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
        use_bias: 布尔值，该层是否使用偏置向量。
        kernel_initializer: kernel 权值矩阵的初始化器， 用于输入的线性转换 (详见 initializers)。
        recurrent_initializer: recurrent_kernel 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 initializers)。
        bias_initializer:偏置向量的初始化器 (详见initializers).
        kernel_regularizer: 运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        recurrent_regularizer: 运用到 recurrent_kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        recurrent_constraint: 运用到 recurrent_kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 运用到偏置向量的约束函数 (详见 constraints)。
        dropout: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
        recurrent_dropout: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换。
        implementation: 实现模式，1 或 2。 模式 1 将把它的操作结构化为更多的小的点积和加法操作， 
        而模式 2 将把它们分批到更少，更大的操作中。 这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
        return_sequences: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
        return_state: 布尔值。除了输出之外是否返回最后一个状态。
        go_backwards: 布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
        stateful: 布尔值 (默认 False)。 如果为 True，
        则批次中索引 i 处的每个样品的最后状态 将用作下一批次中索引 i 样品的初始状态。
        unroll: 布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，
        但它往往会占用更多的内存。 展开只适用于短序列。
        reset_after:
        GRU 公约 (是否在矩阵乘法之前或者之后使用重置门)。 False =「之前」(默认)，Ture =「之后」( CuDNN 兼容)。
        '''

        self.gru_size = gru_size
    def call(self,sequence,states):
        embed = self.embedding(sequence)
        # output,state_h,context_v = self.gru(embed,initial_state=states)
        output, state_h = self.gru(embed, initial_state=states)
        # states:hidden
        return output,state_h
    def init_states(self,batch_size:int):
        return tf.zeros([batch_size,self.gru_size])


class BahdanauAttention(tf.keras.Model):
    # other attention is LuongAttention
    def __init__(self, units):
        # unit: gru_size
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: hidden
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
        gru_out, state_h = self.gru(embed, initial_state=state)
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

    def init_states(self,batch_size):
        return (tf.zeros([batch_size,self.gru_size]),
                tf.zeros([batch_size,self.gru_size]))

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
        output,state_h,context_v = self.encoder(example_input_batch, sample_hidden)
        #attention part
        self.attention_layer = BahdanauAttention(2)
        attention_result, attention_weights = self.attention_layer(state_h, output)
        #decoder part
        self.decoder = Decoder(self.tokenizer_decoder,self.embedding_dim,self.decoder_embedding,self.unit)
        logits, state_h, state_c, aligment  = self.decoder(tf.random.uniform((self.batch_size, 1)), sample_hidden,
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
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden ,context= self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.tokenizer_decoder.word_index['<s>']] * self.batch_size, 1)
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









