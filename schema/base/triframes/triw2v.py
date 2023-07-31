#!/usr/bin/env python

import argparse
import sys
from collections import Counter

import faiss
import networkx as nx
import numpy as np
from chinese_whispers import chinese_whispers, aggregate_clusters

from utils import triples, grouper, word_vectors, words_vec

parser = argparse.ArgumentParser()
parser.add_argument('--neighbors', '-n', type=int, default=10)
parser.add_argument('--min-weight', type=float, default=0.)
parser.add_argument('--pickle', type=argparse.FileType('wb'))
parser.add_argument('triples', type=argparse.FileType('r', encoding='UTF-8'))
group = parser.add_mutually_exclusive_group()
group.add_argument('--w2v', default=None, type=argparse.FileType('rb'))
group.add_argument('--pyro', default=None, type=str)
args = parser.parse_args()

w2v = word_vectors(args, lambda args: parser.error('Please set the --w2v or --pyro option.'))

spos, _ = triples(args.triples, min_weight=args.min_weight, build_index=False)

vocabulary = {word for triple in spos for word in (triple.subject, triple.predicate, triple.object)}

vectors = {}

for words in grouper(vocabulary, 512):
    vectors.update(words_vec(w2v, words))

spos = [triple for triple in spos if
        triple.subject in vectors and triple.predicate in vectors and triple.object in vectors]

# noinspection PyUnboundLocalVariable
X, index2triple = np.empty((len(spos), w2v.vector_size * 3), 'float32'), {}

for i, triple in enumerate(spos):
    X[i] = np.concatenate((vectors[triple.subject], vectors[triple.predicate], vectors[triple.object]))
    index2triple[i] = (triple.subject, triple.predicate, triple.object)

knn = faiss.IndexFlatIP(X.shape[1])
knn.add(X)

G, maximal_distance = nx.Graph(), -1

for slice in grouper(range(X.shape[0]), 2048):
    slice = [j for j in slice if j is not None]

    D, I = knn.search(X[slice, :], args.neighbors + 1)

    last = min(slice)
    print('%d / %d' % (last, X.shape[0]), file=sys.stderr)

    for i, (_D, _I) in enumerate(zip(D, I)):
        source = index2triple[last + i]
        words = Counter()

        for d, j in zip(_D.ravel(), _I.ravel()):
            if last + i != j:
                words[index2triple[j]] = float(d)

        for target, distance in words.most_common(args.neighbors):
            G.add_edge(source, target, weight=distance)
            maximal_distance = distance if distance > maximal_distance else maximal_distance

for _, _, d in G.edges(data=True):
    d['weight'] = maximal_distance / d['weight']

if args.pickle is not None:
    import pickle

    pickle.dump(list(G.edges(data=True)), args.pickle, protocol=3)
    sys.exit(0)

chinese_whispers(G, weighting='top', iterations=20)
clusters = aggregate_clusters(G)

for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
    print('# Cluster %d\n' % label)

    predicates = {predicate for predicate, _, _ in cluster}
    subjects = {subject for _, subject, _ in cluster}
    objects = {object for _, _, object in cluster}

    print('Predicates: %s' % ', '.join(predicates))
    print('Subjects: %s' % ', '.join(subjects))
    print('Objects: %s\n' % ', '.join(objects))
