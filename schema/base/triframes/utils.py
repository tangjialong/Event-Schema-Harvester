from collections import namedtuple, defaultdict
from itertools import zip_longest


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


Triple = namedtuple('Triple', 'subject predicate object weight')


def triples(f, min_weight=None, build_index=True):
    spos, index = [], defaultdict(set)

    for line in f:
        predicate, subject, object, weight = line.strip().split('\t', 3)
        predicate, *tail = predicate.rpartition('#')

        if not predicate:
            predicate = tail[-1]

        subject, *tail = subject.rpartition('#')

        if not subject:
            subject = tail[-1]

        object, *tail = object.rpartition('#')

        if not object:
            object = tail[-1]

        weight = float(weight)

        if (min_weight is not None and weight < min_weight) or not subject or not predicate or not object:
            continue

        spos.append(Triple(subject, predicate, object, weight))

        if build_index:
            index[predicate].add(len(spos) - 1)

    return spos, index


def word_vectors(args, fallback=lambda x: None):
    if args.w2v:
        from gensim.models import KeyedVectors
        w2v = KeyedVectors.load_word2vec_format(args.w2v, binary=True, unicode_errors='ignore')
        w2v.init_sims(replace=True)
        return w2v
    elif args.pyro:
        import Pyro4
        Pyro4.config.SERIALIZER = 'pickle'
        w2v = Pyro4.Proxy(args.pyro)
        return w2v
    else:
        return fallback(args)


def words_vec(w2v, words, use_norm=False):
    """
    Return a dict that maps the given words to their embeddings.
    """
    if callable(getattr(w2v, 'words_vec', None)):
        return w2v.words_vec(words, use_norm)

    return {word: w2v.wv.word_vec(word, use_norm) for word in words if word in w2v.wv}
