import codecs
from pandas import read_csv
from jnt.pcz.sense_clusters import SenseClusters
from time import time
import operator
from traceback import format_exc
import sys


SKIP_POS = set(["-lrb-", "-LRB-", "-RRB-","-rrb-", "-lsb-", "-LSB-", "-rsb-", "-RSB-", "SYM", "SY"])
FUNCTIONAL_LEMMAS = set(["whom","that","which","where","what"]) 
FUNCTIONAL_POS = set(["PRP","PR","FW","DT","DET","CC","CONJ","CD","NUM","WH","WP"])
SKIP_HYPERS = set(["form","member","condition","task","service","sector", "topic", "source","part","modification","feature", "thing", "object", "", "stuff", "item","issue", "information", "medium", "tag", "work","material"])


min_common = 20
verbose = False 
max_nn = 5


err_f_num = 0
f_num = 0


def skip(subj_lemma, subj_pos, obj_lemma, obj_pos, one_stopword_is_ok=False):
    """ Returns true if the triple should be skipped on the basis 
    of the fact that a lemma/pos of its subject/object is in a stoplist. """

    skip_pos_subj = (
            subj_lemma in SKIP_POS or # sometimes lemma is replaced with pos
            subj_pos in SKIP_POS)

    skip_pos_obj = (
            obj_lemma in SKIP_POS or # sometimes lemma is replaced with pos
            obj_pos in SKIP_POS)
    
    skip_lemma_subj = (subj_lemma in FUNCTIONAL_LEMMAS or subj_pos in FUNCTIONAL_POS)
    skip_lemma_obj = (obj_lemma in FUNCTIONAL_LEMMAS or obj_pos in FUNCTIONAL_POS)

    if one_stopword_is_ok:
        skip_pos = skip_pos_subj and skip_pos_obj
        skip_lemma = skip_lemma_subj and skip_lemma_obj
    else:
        skip_pos = skip_pos_subj or skip_pos_obj
        skip_lemma = skip_lemma_subj or skip_lemma_obj

    return skip_pos or skip_lemma


def generalize_by_pos(lemma, pos):
    """ Takes a lemma and a POS and generalizes it if possible. """

    if pos in ["PRP", "PR", "FW"]:
        if lemma.lower() == "it" or lemma.lower() == "its": s = "THING"
        else: s = "PERSON"
    elif pos in ["DT", "DET"]: s = "DET"
    elif pos in ["CC", "CONJ"]: s = "CONJ"
    elif pos in ["CD", "NU", "NUM"]: s = "NUM"
    else: s = lemma
    
    if lemma == "who": s = "PERSON"
        
    return s 


def parse_subj_obj(f):
    """ Performs parsing of a string like 'it#PR_woman#NN:10.0'. """

    subj_obj, score = f.split(":")
    score = float(score)
    subj, obj = subj_obj.split("_")
    subj_lemma, subj_pos = subj.split("#")
    obj_lemma, obj_pos = obj.split("#")

    return (subj_lemma, subj_pos, obj_lemma, obj_pos, score)


def parse_features(features_str):    
    """ Parses a string of verb features e.g. at#WD_VALID#NN:5.0, at#WD_CENTER#NN:5.0
    into a structured representation (a set). """

    global err_f_num 
    global f_num 
    
    features = [f.strip() for f in features_str.split(",")]
    res = set()
    
    for f in features:
        try:
            f_num += 1 
            subj_lemma, subj_pos, obj_lemma, obj_pos, score = parse_subj_obj(f)
            
            if skip(subj_lemma, subj_pos, obj_lemma, obj_pos):
                if verbose: print("Warning: skipping feature '{}'".format(f))
                err_f_num += 1
                continue
            s = generalize_by_pos(subj_lemma, subj_pos)
            o = generalize_by_pos(obj_lemma, obj_pos)
            res.add(s + "_" + o)
            res.add(s + "_" + o)
        except:
            if verbose: print("Warning: bad feature '{}'".format(f))
            err_f_num += 1
            
    return res
 

def featurize(output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as out:
        inventory_voc = set(sc.data)
        features_voc = set(features)
        common_voc = inventory_voc.intersection(features_voc)
        print("Vocabulary of verbs (full and common with inv.):", len(features_voc), len(common_voc))

        for verb in common_voc:
            for sense_id in sc.data[verb]:
                verb_features = features[verb]
                used_rverbs = [(verb + "#" + unicode(int(float(sense_id))), len(verb_features))]
                c = sc.data[verb][sense_id]["cluster"]
                i = 0
                for rverb_sense, _ in sorted(c.items(), key=operator.itemgetter(1), reverse=True)[:max_nn]:
                    try:
                        i += 1
                        rverb, pos, rsense_id = rverb_sense.split("#")
                        rverb_pos = rverb + "#VB"
                        if rverb_pos not in features:
                            rverb_pos = rverb.lower() + "#VB"
                            if rverb_pos not in features:
                                if verbose: print("Warning: related verb '{}' is OOV.".format(rverb_sense))
                                continue
                        rverb_features = features[rverb_pos]
                        
                        common_features = verb_features.intersection(rverb_features)

                        if len(common_features) >= min_common:
                            used_rverbs.append((rverb_sense, len(common_features)))
                            verb_features = common_features
                        else:
                            break
                    except KeyboardInterrupt:
                        break
                    except:
                        if verbose: print("Warning: bad related verb '{}'".format(rverb_sense))
                        print(format_exc())
                
                if i > 0 and len(used_rverbs) > 1 and len(verb_features) > 0:
                    out.write("{}\t{}\n".format(
                            ", ".join("{}:{}".format(w, f) for w, f in used_rverbs), 
                            ", ".join(verb_features)))


def compute_exact_full(wf_w_fpath, pcz_fpath, output_fpath):
    tic = time()
    wfw = read_csv(wf_w_fpath, sep="\t", encoding="utf-8", names=["verb", "features"])
    sc = SenseClusters(pcz_fpath, strip_dst_senses=False, load_sim=True)
    features = {row.verb: parse_features(row.features) for i, row in wfw.iterrows()}
    print("Loaded featrues in", time() - tic, "sec.")
    print("Number of skipped features:", err_f_num, "of", f_num)

    tic = time()
    featurize(output_fpath)
    print("Featurized in", time() - tic, "sec.")
    sys.stdout.flush()


# example:
#compute_exact_full(
#    "/home/panchenko/tmp/verbs/2.6t/wf-w.csv.gz",
#    "/home/panchenko/tmp/originalinventories/wiki-n30.csv.gz",
#    "/home/panchenko/tmp/verbs/exact-full.txt")

