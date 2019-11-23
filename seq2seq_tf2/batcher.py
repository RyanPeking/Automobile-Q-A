import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'data_process'))
from data import input_to_ids, report_to_ids
import pandas as pd


def pad_decoder_inp_targ(dec_input, target, max_len, pad_id):
    """Pad decoder input and target sequences with pad_id up to max_len."""
    while len(dec_input) < max_len:
        dec_input.append(pad_id)
    while len(target) < max_len:
        target.append(pad_id)
    return dec_input, target


def pad_encoder_input(enc_input, enc_input_extend_vocab, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(enc_input) < max_len:
        enc_input.append(pad_id)
    while len(enc_input_extend_vocab) < max_len:
        enc_input_extend_vocab.append(pad_id)
    return enc_input, enc_input_extend_vocab


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


def example_generator(filename, vocab, max_enc_len, max_dec_len, batch_size, mode):
    data = pd.read_csv(filename)

    if mode == "train":
        buffer_size = len(data['input'])
        parser_dataset = tf.data.Dataset.from_tensor_slices((data['input'], data['Report'])).shuffle(buffer_size)

    elif mode == "test" or mode == "eval":
        parser_dataset = tf.data.Dataset.from_tensor_slices(data['input'])

    parser_dataset = parser_dataset.batch(batch_size=batch_size, drop_remainder=True)

    for raw_record in parser_dataset:
        start_decoding = vocab.word_to_id(vocab.START_DECODING)
        stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)
        pad_decoding = vocab.word_to_id(vocab.PAD_TOKEN)

        input = raw_record[0].numpy()[0].decode('UTF-8')
        input_words = input.split()[:max_enc_len]
        enc_len = len(input_words)
        enc_input = [vocab.word_to_id(w) for w in input_words]
        enc_input_extend_vocab, input_oovs = input_to_ids(input_words, vocab)
        enc_input, enc_input_extend_vocab = pad_decoder_inp_targ(enc_input, enc_input_extend_vocab, max_enc_len, pad_decoding)

        if mode == "train" or mode == "eval":
            report = raw_record[1].numpy()[0].decode('UTF-8')
            report_words = report.split()
            report_ids = [vocab.word_to_id(w) for w in report_words]
            report_ids_extend_vocab = report_to_ids(report_words, vocab, input_oovs)
            dec_input, target = get_dec_inp_targ_seqs(report_ids, max_dec_len, start_decoding, stop_decoding)
            _, target = get_dec_inp_targ_seqs(report_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
            dec_len = len(dec_input)
            dec_input, target = pad_decoder_inp_targ(dec_input, target, max_dec_len, pad_decoding)

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "input_oovs": input_oovs,
                "dec_input": dec_input,
                "target": target,
                "dec_len": dec_len,
                "input": input,
                "report": report,
            }
        elif mode == "test":
            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "input_oovs": input_oovs,
                "input": input,
            }

        if mode == "test" or mode == "eval":
            for _ in range(batch_size):
                yield output
        else:
            yield output


def batch_generator(generator, filename, vocab, max_enc_len, max_dec_len, batch_size, mode):
    dataset = tf.data.Dataset.from_generator(lambda: generator(filename, vocab, max_enc_len, max_dec_len, batch_size, mode),
                                             output_types={
                                                 "enc_len": tf.int32,
                                                 "enc_input": tf.int32,
                                                 "enc_input_extend_vocab": tf.int32,
                                                 "input_oovs": tf.string,
                                                 "dec_input": tf.int32,
                                                 "target": tf.int32,
                                                 "dec_len": tf.int32,
                                                 "input": tf.string,
                                                 "report": tf.string,
                                             },
                                             output_shapes={
                                                 "enc_len": [],
                                                 "enc_input": [None],
                                                 "enc_input_extend_vocab": [None],
                                                 "input_oovs": [None],
                                                 "dec_input": [None],
                                                 "target": [None],
                                                 "dec_len": [],
                                                 "input": [],
                                                 "report": [],
                                             })

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=({"enc_len": [],
                                                   "enc_input": [None],
                                                   "enc_input_extend_vocab": [None],
                                                   "input_oovs": [None],
                                                   "dec_input": [max_dec_len],
                                                   "target": [max_dec_len],
                                                   "dec_len": [],
                                                   "input": [],
                                                   "report": []}),
                                   padding_values={"enc_len": -1,
                                                   "enc_input": 1,
                                                   "enc_input_extend_vocab": 1,
                                                   "input_oovs": b'',
                                                   "dec_input": 1,
                                                   "target": 1,
                                                   "dec_len": -1,
                                                   "input": b"",
                                                   "report": b""},
                                   drop_remainder=True)

    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "input_oovs": entry["input_oovs"],
                 "enc_len": entry["enc_len"],
                 "input": entry["input"],
                 "max_oov_len": tf.shape(entry["input_oovs"])[1]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "report": entry["report"]})

    dataset = dataset.map(update)
    return dataset


def batcher(filename, vocab, hps):
    # hps: hyperparameters
    dataset = batch_generator(example_generator, filename, vocab, hps["max_enc_len"],
                              hps["max_dec_len"], hps["batch_size"], hps["mode"])
    return dataset


if __name__ == '__main__':
    import os
    import argparse
    train_path = os.path.join(os.path.abspath('../'), 'data', 'treated_train.csv')
    vocab_path = os.path.join(os.path.abspath('../'), 'data', 'words_frequences.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_enc_len", default=500, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=50, help="Decoder input max sequence length", type=int)
    parser.add_argument("--vocab_size", default=50000, help="Vocabulary size", type=int)
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--mode", default='train', help="mode")
    args = parser.parse_args()
    hps = vars(args)
    # print(hps['max_size'])
    b = batcher(train_path, vocab_path, hps)
    print(b)