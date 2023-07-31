#!/usr/bin/env python

import argparse
import gzip
import re
import sys
from collections import Counter
from functools import partial
from itertools import zip_longest

import faiss
import networkx as nx
import numpy as np
from chinese_whispers import chinese_whispers, aggregate_clusters
from gensim.models import KeyedVectors

from utils import grouper

parser = argparse.ArgumentParser()
parser.add_argument('--neighbors', '-n', type=int, default=10)
parser.add_argument('--pickle', type=argparse.FileType('wb'))
parser.add_argument('words', type=argparse.FileType('rb'))
parser.add_argument('contexts', type=argparse.FileType('rb'))
parser.add_argument('relations', type=argparse.FileType('rb'))
parser.add_argument('triples', type=argparse.FileType('rb'))
args = parser.parse_args()

wordmodel = KeyedVectors.load_word2vec_format(args.words, binary=False)
contextmodel = KeyedVectors.load_word2vec_format(args.contexts, binary=False)
relationmodel = KeyedVectors.load_word2vec_format(args.relations, binary=False)

spos = set()

POS = r'\#\w+$'

extract = partial(re.sub, POS, '')

with gzip.open(args.triples) as f:
    for line in f:
        _, verb, subject, object = line.decode('utf-8').strip().split(' ', 3)

        if verb in wordmodel and subject in contextmodel and object in relationmodel:
            spos.add((verb, subject, object))

spos = list(spos)

index2triple = {}
X = np.empty((len(spos), wordmodel.vector_size + contextmodel.vector_size + relationmodel.vector_size), 'float32')

for i, (verb, subject, object) in enumerate(spos):
    # This changes order from VSO to SVO because I use it everywhere.
    j = 0
    X[i, j:j + contextmodel.vector_size] = contextmodel[subject]

    j += contextmodel.vector_size
    X[i, j:j + wordmodel.vector_size] = wordmodel[verb]

    j += wordmodel.vector_size
    X[i, j:j + relationmodel.vector_size] = relationmodel[object]

    index2triple[i] = (extract(subject), extract(verb), extract(object))

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

    subjects = {subject for subject, _, _ in cluster}
    predicates = {predicate for _, predicate, _ in cluster}
    objects = {object for _, _, object in cluster}

    print('Predicates: %s' % ', '.join(predicates))
    print('Subjects: %s' % ', '.join(subjects))
    print('Objects: %s\n' % ', '.join(objects))
