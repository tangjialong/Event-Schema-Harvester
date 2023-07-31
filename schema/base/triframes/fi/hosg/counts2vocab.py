#!/usr/bin/python3

import gzip
from collections import Counter
from docopt import docopt
from representations.matrix_serializer import save_count_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2pmi.py <counts>
    """)

    counts_path = args['<counts>']

    words = Counter()
    contexts = Counter()
    relations = Counter()
    with gzip.open(counts_path) as f:
        for line in f:
            split = line.decode('utf-8').strip().split()
            if len(split) == 4:
                count, word, context, relation = split
            else:
                count, word, context = split
                relation = None
            count = int(count)
            words[word] += count
            contexts[context] += count
            relations[relation] += count

    words = sorted(list(words.items()), key=lambda x_y: x_y[1], reverse=True)
    contexts = sorted(list(contexts.items()), key=lambda x_y1: x_y1[1], reverse=True)
    relations = sorted(list(relations.items()), key=lambda x_y2: x_y2[1], reverse=True)

    save_count_vocabulary(counts_path + '.words.vocab', words)
    save_count_vocabulary(counts_path + '.contexts.vocab', contexts)
    save_count_vocabulary(counts_path + '.relations.vocab', relations)


if __name__ == '__main__':
    main()
