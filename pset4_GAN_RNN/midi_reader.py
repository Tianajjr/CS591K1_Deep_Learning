import sys
sys.path.append('/path/to/Mozarella/')
from mozarella import midi_embedding_generator,generate_midi_from_embeddings
import os
import random
import copy
import re

import six
import numpy as np
import pickle
import random


def read_embedding_pkl(emb_pkl_path):
    all_raw_node = pickle.load(open(emb_pkl_path + 'all_raw_node.pkl','rb'))
    all_raw_duration = pickle.load(open(emb_pkl_path + 'all_raw_duration.pkl','rb'))
    return all_raw_node, all_raw_duration

def read_embedding_from_midi(embeddings):
    node_list = []
    duration_list = []
    for curr in embeddings:
        node_list.append(np.argmax(curr[:-1]))
        duration_list.append(curr[-1])
    return node_list, duration_list

def read_midi_data(midi_data_path):
    all_raw_node = []
    all_raw_duration = []
    for embeddings in midi_embedding_generator(midi_data_path):
        node_list, duration_list = read_embedding_from_midi(embeddings)
        all_raw_node.append(node_list) 
        all_raw_duration.append(duration_list)
        
    return all_raw_node, all_raw_duration

def save_to_mid(pred_seq, dump_path):
    pred_seq = np.array(pred_seq)
    b = np.zeros((len(pred_seq), 88))
    b[np.arange(len(pred_seq)), pred_seq] = 1
    b = b.astype(int)
    b[:,-1] = 100. 
    b = b.tolist()
    generate_midi_from_embeddings(b,path=dump_path)
    
    
class GeneratorRestartHandler(object):
    def __init__(self, gen_func, argv, kwargv):
        self.gen_func = gen_func
        self.argv = copy.copy(argv)
        self.kwargv = copy.copy(kwargv)
        self.local_copy = self.gen_func(*self.argv, **self.kwargv)
    
    def __iter__(self):
        return GeneratorRestartHandler(self.gen_func, self.argv, self.kwargv)
    
    def __next__(self):
        return next(self.local_copy)
    
def restartable(g_func):
    def tmp(*argv, **kwargv):
        return GeneratorRestartHandler(g_func, argv, kwargv)
    
    return tmp

@restartable
def _batch_midi_generator(all_raw_node, all_raw_duration, batch_size=20, seq_size=360):
    n_batches = int(len(all_raw_node)/batch_size)
    for i in range(n_batches):
        batch_list = []
        for j in range(i*batch_size,(i+1)*batch_size):
            start_pos = random.randint(len(all_raw_node[j])-seq_size-1,len(all_raw_node[j])-1)
            batch_x = all_raw_node[j][start_pos:seq_size+start_pos]
            batch_y = all_raw_node[j][seq_size+start_pos]
            batch_d = all_raw_duration[j][start_pos:seq_size+start_pos]
            batch_list.append((batch_x, batch_y,batch_d))
        batch_x_list, batch_y_list, batch_d_list = zip(*batch_list)
        yield np.array(batch_x_list), np.array(batch_y_list), np.array(batch_d_list)
        
def batch_midi_generator_from_midi(midi_data_path, batch_size=20, seq_size=10):
    all_raw_node, all_raw_duration = read_midi_data(midi_data_path)
    gen = _batch_midi_generator(all_raw_node, all_raw_duration, batch_size, seq_size)
    return gen

def batch_midi_generator_from_pkl(emb_pkl_path, batch_size=20, seq_size=10):
    all_raw_node, all_raw_duration = read_embedding_pkl(emb_pkl_path)
    gen = _batch_midi_generator(all_raw_node, all_raw_duration, batch_size, seq_size)
    return gen
