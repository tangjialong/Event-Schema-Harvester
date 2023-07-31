import pickle as pk
import numpy as np
import os
import random
import prettytable as pt
import argparse
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from base.evaluater import calcACC, check_with_bcubed_lib

def count_tup(data_dict):
    tup2count = defaultdict(int)
    for t, i in data_dict['vocab'].items():
        vs, oh = t
        tup2count[(vs, oh)] = data_dict['tuple_freq'][i]
    return tup2count

def agglo(data_dict, id_map, k, output_dir, times=10):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    tup2count = count_tup(data_dict)
    X = np.concatenate((data_dict['vs_emb'], data_dict['oh_emb']), axis=1)
    golden = [id_map.index(v) for k, v in data_dict['event_type'].items()]
    random.seed(1234)
    np.random.seed(1234)
    all_ARI = []
    all_NMI = []
    all_ACC = []
    all_Bcubed_F1 = []
    all_clusters = []
    for time in range(times):
        hier_predicted = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    
        tuple_clusters = [{} for _ in range(k)]
        for i, clus_num in enumerate(hier_predicted):
            tup = data_dict['inv_vocab'][i]
            tuple_clusters[clus_num][tup] = tup2count[tup]
        hier_results = [sorted(tuple_cluster.keys(), key=lambda x: tuple_cluster[x], reverse=True) for tuple_cluster in tuple_clusters]

        ARI = adjusted_rand_score(golden, hier_predicted)
        NMI = normalized_mutual_info_score(golden, hier_predicted)
        ACC = calcACC(golden, hier_predicted)
        Bcubed_F1 = check_with_bcubed_lib(golden, hier_predicted)

        all_ARI.append(ARI)
        all_NMI.append(NMI)
        all_ACC.append(ACC)
        all_Bcubed_F1.append(Bcubed_F1)
        all_clusters.append(len(hier_results))

        with open(os.path.join(output_dir, f'{time}.json'), 'w') as f:
            for i, cluster in enumerate(hier_results):
                result_string = f"Topic {i} ({len(cluster)}): "
                for vo in cluster:
                    result_string += ', ' + ' '.join([vo[0].split('_')[0], vo[1]])
                f.write(result_string + '\n')
            f.write('\n')
            tb = pt.PrettyTable()
            tb.field_names = ['ARI', 'NMI', 'ACC', 'BCubed-F1', 'V-measure']
            tb.add_row([round(ARI*100, 2), round(NMI*100, 2), round(ACC*100, 2), round(Bcubed_F1*100, 2), round(len(hier_results), 2)])
            f.write(str(tb))

    return np.array(all_ARI).mean(), np.array(all_NMI).mean(), np.array(all_ACC).mean(), np.array(all_Bcubed_F1).mean(), np.array(all_clusters).mean(), \
        np.array(all_ARI).std(), np.array(all_NMI).std(), np.array(all_ACC).std(), np.array(all_Bcubed_F1).std(), np.array(all_clusters).std()