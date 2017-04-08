import numpy as np
import os
from os.path import join, basename
import glob
import math
import pickle

data_dir = '/home/brij/Documents/Project/SLSP_code/src/feats'
num_spks = 10

def load_feats(ftype='mfcc'):
	train_feats = []
	train_labels = []
	test_feats = []
	test_labels = []
	print data_dir

	for idx, feat_file in enumerate(glob.glob(join(data_dir, ftype, 'train', '*.npy'))):
		print idx, feat_file
		if idx < num_spks:
			feats = np.load(feat_file)
			sad = np.load(join(data_dir, 'sad', basename(feat_file)))
			feats = feats[np.where(sad == 1)]

			labels = np.zeros((feats.shape[0], num_spks), dtype='int')
			labels[:, idx] = 1

			train_idx = int(math.floor(feats.shape[0] * 0.7))

			if len(train_feats) == 0:
				train_feats = feats[:train_idx]
				train_labels = labels[:train_idx]
			else:
				train_feats = np.concatenate((train_feats, feats[:train_idx]), axis=0)
				train_labels = np.concatenate((train_labels, labels[:train_idx]), axis=0)

			if len(test_feats) == 0:
				test_feats = feats[train_idx:]
				test_labels = labels[train_idx:]
			else:
				test_feats = np.concatenate((test_feats, feats[train_idx:]), axis=0)
				test_labels = np.concatenate((test_labels, labels[train_idx:]), axis=0)

	# Shuffle the feats
	print "Shuffling..."
	ptr = np.random.permutation(len(train_feats))
	#pte = np.random.permutation(len(test_feats))

	return train_feats[ptr], train_labels[ptr], test_feats, test_labels


def load_convae_feats():
	with open("results/conv_ae_spk/train.pkl", 'r') as trnp, open("results/conv_ae_spk/test.pkl", 'r') as tstp:
    		train_data = pickle.load(trnp)
    		test_data = pickle.load(tstp)

    	return train_data[0], train_data[1], test_data[0], test_data[1]