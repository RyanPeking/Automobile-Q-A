import tensorflow as tf
from model import PGN
from train_helper import train_model
from test_helper import beam_decode
from batcher import batcher
import os
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'data_process'))
from data_process.data import Vocab
from tqdm import tqdm
from rouge import Rouge
import pprint

def train(params):
	assert params["mode"].lower() == "train", "change training mode to 'train'"

	tf.compat.v1.logging.info("Building the model ...")
	model = PGN(params)

	print("Creating the vocab ...")
	vocab = Vocab(params["vocab_path"], params["vocab_size"])

	print("Creating the batcher ...")
	b = batcher(params["data_dir"], vocab, params)

	print("Creating the checkpoint manager")
	checkpoint_dir = "{}".format(params["checkpoint_dir"])
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

	ckpt.restore(ckpt_manager.latest_checkpoint)
	if ckpt_manager.latest_checkpoint:
		print("Restored from {}".format(ckpt_manager.latest_checkpoint))
	else:
		print("Initializing from scratch.")

	tf.compat.v1.logging.info("Starting the training ...")
	train_model(model, b, params, ckpt, ckpt_manager, "output.txt")
 

def test(params):
	assert params["mode"].lower() in ["test","eval"], "change training mode to 'test' or 'eval'"
	assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

	tf.compat.v1.logging.info("Building the model ...")
	model = PGN(params)

	print("Creating the vocab ...")
	vocab = Vocab(params["vocab_path"], params["vocab_size"])

	print("Creating the batcher ...")
	b = batcher(params["data_dir"], vocab, params)

	print("Creating the checkpoint manager")
	checkpoint_dir = "{}".format(params["checkpoint_dir"])
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

	path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
	ckpt.restore(path)
	print("Model restored")

	for batch in b:
		yield beam_decode(model, batch, vocab, params)


def test_and_save(params):
	assert params["test_save_dir"], "provide a dir where to save the results"
	gen = test(params)
	with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
		for i in range(params["num_to_test"]):
			trial = next(gen)
			with open(params["test_save_dir"]+"/input_"+str(i)+".txt", "w") as f:
				f.write("input:\n")
				f.write(trial.text)
				f.write("\n\nreport:\n")
				f.write(trial.report)
			pbar.update(1)

def evaluate(params):
	gen = test(params)
	reals = []
	preds = []
	with tqdm(total=params["max_num_to_eval"],position=0, leave=True) as pbar:
		for i in range(params["max_num_to_eval"]):
			trial = next(gen)
			reals.append(trial.real_report)
			preds.append(trial.report)
			pbar.update(1)
	r = Rouge()
	scores = r.get_scores(preds, reals, avg=True)
	print("\n\n")
	pprint.pprint(scores)