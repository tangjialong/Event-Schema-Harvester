import pickle as pk
import numpy as np
from tqdm import tqdm
from itertools import product
import json
import argparse
from collections import defaultdict
import os
import prettytable as pt
import random
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from base.evaluater import check_with_bcubed_lib
SUBJ = '[SUBJ]'

def save_data_for_triframes(data_dict):
    sense_emb = []
    obj_head_emb = []
    vs_vocab = {}
    oh_vocab = {}
    vs_inv_vocab = []
    oh_inv_vocab = []
    tup2count = defaultdict(int)
    for t, i in data_dict['vocab'].items():
        vs, oh = t
        if vs not in vs_vocab:
            vs_vocab[vs] = len(vs_vocab)
            vs_inv_vocab.append(vs)
            sense_emb.append(data_dict['vs_emb'][i])
        if oh not in oh_vocab:
            oh_vocab[oh] = len(oh_vocab)
            oh_inv_vocab.append(oh)
            obj_head_emb.append(data_dict['oh_emb'][i])
        tup2count[(vs, oh)] = data_dict['tuple_freq'][i]
    sense_emb = np.vstack(sense_emb)
    obj_head_emb = np.vstack(obj_head_emb)
    
    with open('base/triframes/triplets.tsv', 'w') as f:
        for t, i in data_dict['vocab'].items():
            vs, oh = t
            print(f'{vs}\t{SUBJ}\t{oh}\t1.0', file=f)

    dim = sense_emb.shape[1]
    kv = KeyedVectors(dim)
    kv.add(vs_inv_vocab, sense_emb)
    kv.add(oh_inv_vocab, obj_head_emb)
    kv.add([SUBJ], [np.ones(dim)*1e-8])
    kv.save_word2vec_format('base/triframes/w2v.bin', binary=True)
    
    return tup2count

def triframes_call(watset):
    os.chdir('./base/triframes')
    if watset:
        os.system(f'WEIGHT=0 W2V=w2v.bin VSO=triplets.tsv make triw2v-watset.txt')
    else:
        os.system(f'WEIGHT=0 W2V=w2v.bin VSO=triplets.tsv make triw2v.txt')
    os.chdir('../../')
        
def load_triframes(data_dict, watset):
    p2o2c = defaultdict(lambda:defaultdict(int))
    if watset:
        result_file = './base/triframes/triw2v-watset.txt'
        pred_key = 'Predicates'
    else:
        pred_key = 'Subjects'
        result_file = './base/triframes/triw2v.txt'
    with open(result_file) as f:
        curr_c = 0
        preds = None
        objs = None
        for line in f:
            if line.startswith('#'):
                curr_c += 1
                preds = None
                objs = None
            elif line.startswith(pred_key):
                preds = line[len(f'{pred_key}: '):].strip().split(', ')
            elif line.startswith('Objects'):
                objs = line[len('Objects: '):].strip().split(', ')
                for p, o in product(preds, objs):
                    p2o2c[p][o] = curr_c - 1
    triframe_predicted = []
    for t, i in data_dict['vocab'].items():
        vs, oh = t
        pred = p2o2c[vs][oh]
        triframe_predicted.append(pred)
    return triframe_predicted, curr_c

def triframes(watset, data_dict, id_map, output_dir, times=10):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    tup2count = save_data_for_triframes(data_dict)
    golden = [id_map.index(v) for k, v in data_dict['event_type'].items()]
    random.seed(1234)
    np.random.seed(1234)

    all_ARI = []
    all_NMI = []
    all_ACC = []
    all_Bcubed_F1 = []
    all_clusters = []
    for time in range(times):
        triframes_call(watset)
        predicted, num_clus = load_triframes(data_dict, watset)

        ARI = adjusted_rand_score(golden, predicted)
        NMI = normalized_mutual_info_score(golden, predicted)
        # ACC = calcACC(golden, predicted)
        ACC = 0
        Bcubed_F1 = check_with_bcubed_lib(golden, predicted)

        results = [[] for _ in range(num_clus)]
        for i, cluster_id in enumerate(predicted):
            results[cluster_id].append(data_dict['inv_vocab'][i])

        all_ARI.append(ARI)
        all_NMI.append(NMI)
        all_ACC.append(ACC)
        all_Bcubed_F1.append(Bcubed_F1)
        all_clusters.append(len(results))

        with open(os.path.join(output_dir, f'{time}.json'), 'w') as f:
            for i, cluster in enumerate(results):
                result_string = f"Topic {i} ({len(cluster)}): "
                for vo in cluster:
                    result_string += ', ' + ' '.join([vo[0].split('_')[0], vo[1]])
                f.write(result_string + '\n')
            f.write('\n')
            tb = pt.PrettyTable()
            tb.field_names = ['ARI', 'NMI', 'ACC', 'BCubed-F1', 'V-measure']
            tb.add_row([round(ARI*100, 2), round(NMI*100, 2), round(ACC*100, 2), round(Bcubed_F1*100, 2), round(len(results), 2)])
            f.write(str(tb))
    
    return np.array(all_ARI).mean(), np.array(all_NMI).mean(), np.array(all_ACC).mean(), np.array(all_Bcubed_F1).mean(), np.array(all_clusters).mean(), \
        np.array(all_ARI).std(), np.array(all_NMI).std(), np.array(all_ACC).std(), np.array(all_Bcubed_F1).std(), np.array(all_clusters).std()