#!/usr/bin/env python

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--no-header', dest='header', action='store_false')
parser.add_argument('pickle', type=argparse.FileType('rb'))
args = parser.parse_args()

data = pickle.load(args.pickle)

if args.header:
    print('\t'.join(('source', 'target', 'weight')))

for edge in data:
    source = '|'.join(edge[0])
    target = '|'.join(edge[1])
    weight = str(edge[2]['weight'])

    print('\t'.join((source, target, weight)))
