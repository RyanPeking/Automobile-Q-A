import tensorflow as tf
from tranformer_layers import EncoderLayer,DecoderLayer
from transformer_utils import positional_encoding

class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,
                 maximum_position_encoding,rate=0.1):
        super(Encoder,self).__init__()

        self.d_model=d_model
        self.num_layers=num_layers
        self.embedding=tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.pos_encoding=positional_encoding(maximum_position_encoding,self.d_model)
        self.enc_layers=[EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
        self.dropout=tf.keras.layers.Dropout(rate)

    def call(self,x,training,mask):
        seq_len=tf.shape(x)[1]
        x=self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x+=self.pos_encoding[:,:seq_len,:]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x=self.enc_layers[i](x,training,mask)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,target_vocab_size,
                 maximum_position_encoding,rate=0.1):
        super(Decoder,self).__init__()
        self.num_layers=num_layers
        self.d_model=d_model

        self.embedding=tf.keras.layers.Embedding(target_vocab_size,d_model)
        self.pos_encoding=positional_encoding(maximum_position_encoding,d_model)

        self.dec_layers=[DecoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]

        self.dropout=tf.keras.layers.Dropout(rate)

    def call(self,x,enc_output,training,look_ahead_mask,padding_mask):
        attention_weights={}
        seq_len=tf.shape(x)[1]
        embed_x =self.embedding(x)
        x = embed_x*tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        out = self.dropout(x, training=training)

        for i in range(self.num_layers):
            out,block1,block2,context_vector=self.dec_layers[i](x,enc_output,training,look_ahead_mask,padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return out,attention_weights,embed_x,context_vector

class Pointer(tf.keras.layers.Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, state, dec_inp):
        pointer = tf.nn.sigmoid(self.w_c_reduce(context_vector) + self.w_s_reduce(state) + self.w_i_reduce(dec_inp))
        return pointer


class Transformer(tf.keras.Model):
    def __init__(self,params):

        super(Transformer,self).__init__()

        self.num_layers=params["num_layers"]
        self.num_heads=params["num_heads"]
        self.vocab_size=params["vocab_size"]
        self.batch_size=params["batch_size"]
        self.d_model=params["d_model"]
        self.depth=self.d_model//self.num_heads


        self.encoder=Encoder(params["num_layers"], params["d_model"], params["num_heads"], params["dff"],
                           params["vocab_size"], params["pe_input"], params["rate"])
        self.decoder=Decoder(params["num_layers"], params["d_model"], params["num_heads"], params["dff"],
                           params["vocab_size"], params["pe_target"], params["rate"])

        self.final_layer=tf.keras.layers.Dense(params["vocab_size"])
        self.pointer=Pointer()
        self.mode=params["mode"]

    def call(self,inp,tar,enc_padding_mask,look_ahead_mask,dec_padding_mask,extended_inp,max_oov_len):
        if self.mode=="train":
            training=True
        else:
            training = False
        enc_output=self.encoder(inp,training,enc_padding_mask)


        # attention_output shape=(batch_size, num_heads, seq_len_q, depth)
        # attention_weights shape=(batch_size, num_heads, seq_len_q, seq_len_k)
        dec_output,attention_weights,embed,_=self.decoder(tar,enc_output,training,look_ahead_mask,dec_padding_mask)

        attn_dists = attention_weights[
            'decoder_layer{}_block2'.format(self.num_layers)]  # (batch_size,num_heads, targ_seq_len, inp_seq_len)

        # context vectors
        enc_out_shape = tf.shape(enc_output)
        context = tf.reshape(enc_output, (enc_out_shape[0], enc_out_shape[1], self.num_heads,
                                          self.depth))  # shape : (batch_size, input_seq_len, num_heads, depth)
        context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, num_heads, input_seq_len, depth)
        context = tf.expand_dims(context, axis=2)  # (batch_size, num_heads, 1, input_seq_len, depth)

        attn = tf.expand_dims(attn_dists, axis=-1)  # (batch_size, num_heads, target_seq_len, input_seq_len, 1)

        context = context * attn  # (batch_size, num_heads, target_seq_len, input_seq_len, depth)
        context = tf.reduce_sum(context, axis=3)  # (batch_size, num_heads, target_seq_len, depth)
        context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, target_seq_len, num_heads, depth)
        context = tf.reshape(context, (
        tf.shape(context)[0], tf.shape(context)[1], self.d_model))  # (batch_size, target_seq_len, d_model)


        p_gens=self.pointer(context, dec_output, embed)


        final_output=self.final_layer(dec_output)


        attn_dists = tf.reduce_sum(attn_dists, axis=1) / self.num_heads  # (batch_size, targ_seq_len, inp_seq_len)


        final_dists = self._calc_final_dist(extended_inp,
                                            tf.unstack(final_output, axis=1),
                                            tf.unstack(attn_dists, axis=1),
                                            tf.unstack(p_gens, axis=1),
                                            max_oov_len,
                                            self.vocab_size,
                                            self.batch_size)
        final_output = tf.stack(final_dists, axis=1)

        return final_output,attn_dists

    def _calc_final_dist(self,_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov, vocab_size,
                         batch_size):


        """
        Calculate the final distribution, for the pointer-generator model
        Args:
        vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                    The words are in the order they appear in the vocabulary file.
        attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
        Returns:
        final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        # _enc_batch_extend_vocab = [1,......,500]
        # 没有加gen，如果input输入含有oov 的词，那我们将其ID 设为1
        # _enc_batch_extend_vocab 是将oov 的词改为超过词表的ID来标志，而不是用统一的1来标志

        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
        #词表的分布，可以理解伟先验的分布
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens,attn_dists)]
        #attetion生成的分布

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        batch_oov_len=tf.reduce_sum(batch_oov)
        extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
        #本身的词表长度再加上预设的oov词长度
        extra_zeros = tf.zeros((batch_size, batch_oov_len))
        # list length max_dec_steps of shape (batch_size, extended_vsize)
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
        # then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over句子的长度
        batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
        indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, attn_len, 2)
        shape = [batch_size, extended_vsize]
        # list length max_dec_steps (batch_size, extended_vsize)
        # copy_dist shape=(batch_size, max_length, 1)
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
        # the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                       zip(vocab_dists_extended, attn_dists_projected)]

        return final_dists