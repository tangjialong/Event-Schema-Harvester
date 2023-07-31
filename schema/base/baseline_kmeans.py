import pickle as pk
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as euc
import argparse
import random
import os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import prettytable as pt

from base.evaluater import calcACC, check_with_bcubed_lib

def kmeans(data_dict, id_map, k, output_dir, times=10):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
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
        kmeans = KMeans(n_clusters=k).fit(X)
        centers = kmeans.cluster_centers_
        predicted = kmeans.labels_
        
        mention_clusters = [set() for _ in range(k)]
        mention_clusters_tup_id = [[] for _ in range(k)]
        for i, clus_num in enumerate(predicted):
            tup = data_dict['inv_vocab'][i]
            if tup not in mention_clusters[clus_num]:
                mention_clusters[clus_num].add(tup)
                mention_clusters_tup_id[clus_num].append(i)
        kmeans_results = []
        for tup_id, center in zip(mention_clusters_tup_id, centers):
            dist = euc(center[np.newaxis, :], X[tup_id])[0]
            ranked = [data_dict['inv_vocab'][tup_id[i]] for i in np.argsort(dist)]
            kmeans_results.append(ranked)
        
        ARI = adjusted_rand_score(golden, predicted)
        NMI = normalized_mutual_info_score(golden, predicted)
        ACC = calcACC(golden, predicted)
        Bcubed_F1 = check_with_bcubed_lib(golden, predicted)

        all_ARI.append(ARI)
        all_NMI.append(NMI)
        all_ACC.append(ACC)
        all_Bcubed_F1.append(Bcubed_F1)
        all_clusters.append(len(kmeans_results))

        with open(os.path.join(output_dir, f'{time}.json'), 'w') as f:
            for i, cluster in enumerate(kmeans_results):
                result_string = f"Topic {i} ({len(cluster)}): "
                for vo in cluster:
                    result_string += ', ' + ' '.join([vo[0].split('_')[0], vo[1]])
                f.write(result_string + '\n')
            f.write('\n')
            tb = pt.PrettyTable()
            tb.field_names = ['ARI', 'NMI', 'ACC', 'BCubed-F1', 'V-measure']
            tb.add_row([round(ARI*100, 2), round(NMI*100, 2), round(ACC*100, 2), round(Bcubed_F1*100, 2), round(len(kmeans_results), 2)])
            f.write(str(tb))
    
    return np.array(all_ARI).mean(), np.array(all_NMI).mean(), np.array(all_ACC).mean(), np.array(all_Bcubed_F1).mean(), np.array(all_clusters).mean(), \
        np.array(all_ARI).std(), np.array(all_NMI).std(), np.array(all_ACC).std(), np.array(all_Bcubed_F1).std(), np.array(all_clusters).std()