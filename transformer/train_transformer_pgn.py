import tensorflow as tf
# tf.enable_eager_execution()
import time



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,d_model,warmup_steps=4000):
        super(CustomSchedule,self).__init__()

        self.d_model=d_model
        self.d_model=tf.cast(self.d_model,tf.float32)

        self.warmup_steps=warmup_steps

    def __call__(self,step):
        arg1=tf.math.rsqrt(step)
        arg2=step*(self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)


def training(model, dataset, params, ckpt, manager):

    loss=0
    learning_rate = CustomSchedule(params["d_model"])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def single_training(input_sequence, enc_extended_inp, target_sequence, target_real_sequence, batch_oov_len):

        def loss_fuction(real, pred):
            # real shape=(batch_size,length)
            mask = tf.math.logical_not(tf.math.equal(real, 0))  # True or False
            loss = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss.dtype)
            loss *= mask

            return tf.reduce_mean(loss), mask

        def create_padding_mask(seq):

            # seq shape=(batch_size,seq_len)
            seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

            return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size,1,1,seq_len)

        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return mask  # shape=(seq_len,seq_len) 上三角

        def create_masks(inp, tar):
            enc_padding_mask = create_padding_mask(inp)
            dec_padding_mask = create_padding_mask(inp)  # enc_dec_attention_bias

            look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
            dec_target_padding_mask = create_padding_mask(tar)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

            return enc_padding_mask, combined_mask, dec_padding_mask

        with tf.GradientTape() as tape:
            enc_padding_mask, look_ahead_mask, dec_padding_mask=create_masks(input_sequence, target_sequence)

            final_dist, attentions = model(input_sequence, target_sequence, enc_padding_mask,look_ahead_mask,dec_padding_mask,enc_extended_inp,
                                                    batch_oov_len)
            batch_loss, _ = loss_fuction(target_real_sequence, final_dist)

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return batch_loss

    for batch in dataset:
        start = time.time()
        batch_loss=single_training(batch[0]["enc_input"],
                                   batch[0]["extended_enc_input"],
                                   batch[1]["dec_input"],
                                   batch[1]["dec_target"],
                                   batch[0]["max_oov_len"])
        loss += batch_loss
        if int(ckpt.step) % 10 == 0:
            print('Step {} Loss {:.4f} Time {}'.format(int(ckpt.step), batch_loss.numpy(),time.time()-start))


        if int(ckpt.step) == params["max_steps"]:
            manager.save(checkpoint_number=int(ckpt.step))
            print("Saved checkpoint for step {}".format(int(ckpt.step)))
            break
        if int(ckpt.step) % params["checkpoints_save_steps"] == 0:
            manager.save(checkpoint_number=int(ckpt.step))
            print("Saved checkpoint for step {}".format(int(ckpt.step)))
        ckpt.step.assign_add(1)
