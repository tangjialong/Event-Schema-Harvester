#!/usr/bin/python3

import sys
import os
import gzip
import numpy as np
import gensim
import logging
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import pylab as plot


def embed(contexts, vec_matrix, cluster_labels, goldclusters=None):
    if goldclusters is None:
        goldclusters = [0] * len(cluster_labels)
    embedding = PCA(n_components=2)
    y = embedding.fit_transform(vec_matrix)
    xpositions = y[:, 0]
    ypositions = y[:, 1]
    plot.clf()
    colors = ['black', 'cyan', 'red', 'lime', 'brown', 'yellow', 'magenta', 'goldenrod', 'navy', 'purple', 'silver']
    markers = ['.', 'o', '*', '+', 'x', 'D']
    for label, x, y, cluster_label, goldcluster in zip(contexts, xpositions, ypositions, cluster_labels, goldclusters):
        plot.scatter(x, y, 20, marker=markers[int(float(goldcluster))], color=colors[cluster_label])
        # plot.annotate(label, xy=(x, y), size='small', color=colors[cluster_label])

    plot.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plot.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plot.legend(loc='best')
    plot.savefig('pca.png', dpi=300)
    plot.close()
    plot.clf()
    return y


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

directory2process = sys.argv[1]
n_clusters = int(sys.argv[2])

contextfile = os.path.join(directory2process, 'context-matrix')
wordfile = os.path.join(directory2process, 'word-matrix')
relationfile = os.path.join(directory2process, 'relation-matrix')
triplesfile = os.path.join(directory2process, 'counts.gz')

contextmodel = gensim.models.KeyedVectors.load_word2vec_format(contextfile, binary=False)
wordmodel = gensim.models.KeyedVectors.load_word2vec_format(wordfile, binary=False)
relationmodel = gensim.models.KeyedVectors.load_word2vec_format(relationfile, binary=False)

triples = set()

triplesdata = gzip.open(triplesfile)

for line in triplesdata:
    res = line.decode('utf-8').strip().split()
    triples.add((res[1], res[2], res[3]))

triples = list(triples)

matrix = np.empty((len(triples), wordmodel.vector_size * 3))

counter = 0
missing = 0
for triple in triples:
    (word, context, relation) = triple
    if word in wordmodel and context in contextmodel and relation in relationmodel:
        wordvector = wordmodel[word]
        contextvector = contextmodel[context]
        relationvector = relationmodel[relation]
    else:
        missing += 1
        continue
    conc_vector = np.concatenate((wordvector, contextvector, relationvector))
    matrix[counter, :] = conc_vector
    counter += 1

print('Skipped %d triples' % missing, file=sys.stderr)
print('Final matrix shape:', matrix.shape, file=sys.stderr)

clustering = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=8).fit(matrix)
predicted = clustering.labels_.tolist()

clusters = {}

for triple, p_cluster in zip(triples, predicted):
    (predicate, obj, subject) = triple
    if str(p_cluster) not in clusters:
        clusters[str(p_cluster)] = {"predicates": set(), 'subjects': set(), 'objects': set()}
    clusters[str(p_cluster)]['predicates'].add(predicate.split('#')[0])
    clusters[str(p_cluster)]['subjects'].add(subject.split('#')[0])
    clusters[str(p_cluster)]['objects'].add(obj.split('#')[0])

for cluster in sorted(clusters):
    print('\n# Cluster %s \n' % cluster)
    print('Predicates:', ', '.join(clusters[cluster]['predicates']))
    print('Subjects:', ', '.join(clusters[cluster]['subjects']))
    print('Objects:', ', '.join(clusters[cluster]['objects']))

#  embed(triples, matrix, predicted)
