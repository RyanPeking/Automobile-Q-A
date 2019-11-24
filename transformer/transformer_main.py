import argparse
from train_transformer import train, test_and_save, evaluate
import os
from pretrained_embedding import get_embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_enc_len", default=400, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=50, help="Decoder input max sequence length", type=int)
    parser.add_argument("--max_dec_steps", default=120, help="maximum number of words of the predicted abstract",
                        type=int)
    parser.add_argument("--min_dec_steps", default=30, help="Minimum number of words of the predicted abstract",
                        type=int)
    parser.add_argument("--batch_size", default=64, help="batch size", type=int)
    parser.add_argument("--beam_size", default=4,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--vocab_size", default=50000, help="Input vocabulary size", type=int)
    parser.add_argument("--pe_input", default=50000, help="", type=int)
    parser.add_argument("--pe_target", default=50000, help="", type=int)

    parser.add_argument("--embed_size", default=100, help="Words embeddings dimension", type=int)
    parser.add_argument("--d_model", default=100, help="Model dimension", type=int)
    parser.add_argument("--num_heads", default=10, help="Number of heads", type=int)
    parser.add_argument("--num_layers", default=4, help="Number of layers", type=int)
    parser.add_argument("--dff", default=512, help="Point_wise_feed_forward_network", type=int)

    parser.add_argument("--learning_rate", default=0.15, help="Learning rate", type=float)
    parser.add_argument("--rate", default=0.1, help="Dropout rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer API documentation on tensorflow site for more details.",
                        type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)
    parser.add_argument("--checkpoints_save_steps", default=100, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--max_steps", default=150000, help="Max number of iterations", type=int)
    parser.add_argument("--num_to_test", default=100, help="Number of examples to test", type=int)
    parser.add_argument("--max_num_to_eval", default=5, help="Max number of examples to evaluate", type=int)
    parser.add_argument("--mode", help="training, eval or test options", default="train", type=str)
    parser.add_argument("--model_path", help="Path to a specific model", default="./save/", type=str)
    parser.add_argument("--checkpoint_dir", help="Checkpoint directory", default="./save/", type=str)
    parser.add_argument("--test_save_dir", help="Directory in which we store the decoding results", default="./test_result",
                        type=str)
    parser.add_argument("--data_dir", help="Data Folder", default="./data/train_record", type=str)
    parser.add_argument("--test_dir", help="Data Folder",default="./data/test_record", type=str)
    parser.add_argument("--vocab_path", help="Vocab path", default="./data/vocab_dictionary.txt", type=str)
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)

    parser.add_argument("--w2v_model_path", help="pretrained word2vec model path", default="./data/w2v.model", type=str)

    args = parser.parse_args()
    params = vars(args)


    embedding_matrix=get_embedding(params["embed_size"], params["vocab_path"], params["w2v_model_path"])
    params["embedding_matrix"]=embedding_matrix

    assert params["mode"], "mode is required. train, test or eval option"
    assert params["mode"] in ["train", "test", "eval"], "The mode must be train , test or eval"
    assert os.path.exists(params["data_dir"]), "data_dir doesn't exist"
    assert os.path.isfile(params["vocab_path"]), "vocab_path doesn't exist"

    if params["mode"] == "train":
        train(params)
    elif params["mode"] == "test":
        test_and_save(params)
    elif params["mode"] == "eval":
        evaluate(params)


if __name__ == "__main__":
    main()