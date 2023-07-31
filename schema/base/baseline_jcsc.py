import pickle as pk
import numpy as np
from sklearn.cluster import SpectralClustering
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity as cos
import argparse
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random
import os
import prettytable as pt
from base.evaluater import calcACC, check_with_bcubed_lib

def mask_mat_from_C(C, k):
    C_mask = np.zeros((len(C), len(C)), dtype=int)
    for c in range(k):
        mask_1d = 1 - (C - c).astype(bool)
        mask_2d = np.outer(mask_1d, mask_1d)
        C_mask += mask_2d
    return C_mask

def objective(Ct, Ca, kt, ka, sim_t, sim_a):
    Ct_mask = mask_mat_from_C(Ct, kt)

    Ca_mask = mask_mat_from_C(Ca, ka)

    return np.diagonal(sim_t).sum()/2 + np.sum((1-Ct_mask) * sim_t)/2 + np.sum(Ct_mask * (1-sim_t))/2 + \
           np.diagonal(sim_a).sum()/2 + np.sum((1-Ca_mask) * sim_a)/2 + np.sum(Ca_mask * (1-sim_a))/2

def constraint_mat(edges, num, i, C, k):
    t2c = np.zeros((num, k), dtype=bool)
    for e in edges:
        t2c[e[i], C[e[1-i]]] = True
    m1 = np.repeat(t2c[:, np.newaxis, :], num, axis=1)
    m1T = np.transpose(m1, (1, 0, 2))
    return np.log(1+np.sum(m1&m1T, axis=-1).astype(float) / np.sum(m1|m1T, axis=-1).astype(float))

def clustering_jcsc(emb_t, emb_a, edges, kt_min, kt_max, ka_min, ka_max):
    O_min = np.inf
    Ct = None
    Ca = None
    cos_sim_t = cos(emb_t)
    cos_sim_a = cos(emb_a)
    best_kt = None
    best_ka = None
    best_iter = 0

    for kt, ka in product(range(kt_min, kt_max+1), range(ka_min, ka_max+1)):
        # Clustering with Spectral Clustering
        spectral_t = SpectralClustering(n_clusters=kt, affinity='precomputed', n_init=30).fit(cos_sim_t)
        Ct_curr = spectral_t.labels_

        spectral_a = SpectralClustering(n_clusters=ka, affinity='precomputed', n_init=30).fit(cos_sim_a)
        Ca_curr = spectral_a.labels_

        O_curr = objective(Ct_curr, Ca_curr, kt, ka, cos_sim_t, cos_sim_a)

        if O_curr < O_min:
            O_min = O_curr
            Ct = Ct_curr
            Ca = Ca_curr
            best_kt = kt
            best_ka = ka
            best_iter = -1

        for i in range(10):
            sim_t = cos_sim_t + constraint_mat(edges, emb_t.shape[0], 0, Ca_curr, ka)
            spectral_t = SpectralClustering(n_clusters=kt, affinity='precomputed', n_init=30).fit(sim_t)
            Ct_curr = spectral_t.labels_

            sim_a = cos_sim_a + constraint_mat(edges, emb_a.shape[0], 1, Ct_curr, kt)
            spectral_a = SpectralClustering(n_clusters=ka, affinity='precomputed', n_init=30).fit(sim_a)
            Ca_curr = spectral_a.labels_

            O_curr = objective(Ct_curr, Ca_curr, kt, ka, sim_t, sim_a)

            if O_curr < O_min:
                O_min = O_curr
                Ct = Ct_curr
                Ca = Ca_curr
                best_kt = kt
                best_ka = ka
                best_iter = i

    return O_min, Ct, Ca, best_kt, best_ka, best_iter

def prepare_jcsc(data_dict):
    
    sense_emb = []
    obj_head_emb = []
    edges = []
    vs_vocab = {}
    oh_vocab = {}
    vs_inv_vocab = []
    oh_inv_vocab = []
    edge2count = defaultdict(int)
    trigger2edges = defaultdict(list)
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
        edges.append((vs_vocab[vs], oh_vocab[oh]))
        edge2count[edges[-1]] = data_dict['tuple_freq'][i]
        trigger2edges[vs].append(edges[-1])
    return np.vstack(sense_emb), np.vstack(obj_head_emb), edges, vs_inv_vocab, oh_inv_vocab, edge2count, trigger2edges
    
def jcsc(data_dict, id_map, k, output_dir, times=10):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sense_emb, obj_head_emb, edges, vs_inv_vocab, oh_inv_vocab, edge2count, trigger2edges = prepare_jcsc(data_dict)
    golden = [id_map.index(v) for k, v in data_dict['event_type'].items()]
    random.seed(1234)
    np.random.seed(1234)

    all_ARI = []
    all_NMI = []
    all_ACC = []
    all_Bcubed_F1 = []
    all_clusters = []
    for time in range(times):
        _, Ct,_,_,_,_ = clustering_jcsc(sense_emb, obj_head_emb, edges, k, k, k, k)
        tuple_clusters = [{} for _ in range(k)]
        for vs, clus_num in zip(vs_inv_vocab, Ct):
            tuple_clusters[clus_num].update({(vs, oh_inv_vocab[oh_i]):edge2count[(vs_i, oh_i)] for vs_i, oh_i in trigger2edges[vs]})
        ranked_tuple_clusters = [sorted(tuple_cluster.keys(), key=lambda x: tuple_cluster[x], reverse=True) for tuple_cluster in tuple_clusters]

        predicted = {}
        for i, cluster in enumerate(ranked_tuple_clusters):
            for vo in cluster:
                predicted[data_dict['vocab'][vo]] = i
        predicted = sorted(predicted.items(), key=lambda x: x[0])
        predicted = [p[1] for p in predicted]

        ARI = adjusted_rand_score(golden, predicted)
        NMI = normalized_mutual_info_score(golden, predicted)
        ACC = calcACC(golden, predicted)
        Bcubed_F1 = check_with_bcubed_lib(golden, predicted)

        all_ARI.append(ARI)
        all_NMI.append(NMI)
        all_ACC.append(ACC)
        all_Bcubed_F1.append(Bcubed_F1)
        all_clusters.append(len(ranked_tuple_clusters))

        with open(os.path.join(output_dir, f'{time}.json'), 'w') as f:
            for i, cluster in enumerate(ranked_tuple_clusters):
                result_string = f"Topic {i} ({len(cluster)}): "
                for vo in cluster:
                    result_string += ', ' + ' '.join([vo[0].split('_')[0], vo[1]])
                f.write(result_string + '\n')
            f.write('\n')
            tb = pt.PrettyTable()
            tb.field_names = ['ARI', 'NMI', 'ACC', 'BCubed-F1', 'V-measure']
            tb.add_row([round(ARI*100, 2), round(NMI*100, 2), round(ACC*100, 2), round(Bcubed_F1*100, 2), round(len(ranked_tuple_clusters), 2)])
            f.write(str(tb))
    
    return np.array(all_ARI).mean(), np.array(all_NMI).mean(), np.array(all_ACC).mean(), np.array(all_Bcubed_F1).mean(), np.array(all_clusters).mean(), \
        np.array(all_ARI).std(), np.array(all_NMI).std(), np.array(all_ACC).std(), np.array(all_Bcubed_F1).std(), np.array(all_clusters).std()