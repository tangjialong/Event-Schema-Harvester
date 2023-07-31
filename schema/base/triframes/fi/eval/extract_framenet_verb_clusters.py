# Use framenet to generate the lexical unit clusters of frames

from __future__ import print_function
import argparse
import codecs 
from collections import defaultdict
import sys
import csv
from src.builder import *
from src.ecg_utilities import ECGUtilities as utils
# from src.valence_data import *
from src.hypothesize_constructions import *
from scripts import *


verbose = False


def load_framenet(data_path):
    frame_path = data_path + "frame/"
    relation_path = data_path + "frRelation.xml"
    lu_path = data_path + "lu/"
    fnb = FramenetBuilder(frame_path, relation_path, lu_path)
    fn = fnb.read() 
    fn.build_relations()
    fn.build_typesystem()
    return fn, fnb


def add_children(lus, f_target, f):
    if verbose: print(f.name, ">>>", end="")
    if len(f.children) > 0:
        for c in f.children:
            fc = fn.get_frame(c)
            lus[f_target.name] = lus[f_target.name].union(set(fc.lexicalUnits)) 
            add_children(lus, f_target, fc)
    
    
def get_verbs(lu_set):
    return set([lu for lu in lu_set if ".v" in unicode(lu)])
        
    
def get_framenet_clusters(use_children=False, verbs_only=False):
    lus = defaultdict(lambda: set())
    lus_count = 0
    
    for i, f in enumerate(fn.frames): 
        lus[f.name] = lus[f.name].union(set(f.lexicalUnits))
        
        if use_children:
            if verbose: print("\n")
            add_children(lus, f, f)
            
        if verbs_only:
            lus[f.name] = get_verbs(lus[f.name])
        
        lus_count += len(lus[f.name])
        
    print(len(lus), lus_count)
    return lus


def save_framenet_clusters(framenet_clusters, output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as out: 
        out.write("cid\tsize\tcluster\n")
        for f in framenet_clusters:
            fc = ", ".join([unicode(x)for x in framenet_clusters[f]])
            out.write("%s\t%s\t%s\n".format(f, len(framenet_clusters[f]), fc))
    print("Output:", output_fpath)
            

def run(framenet_dir, output_dir):
    fn, fnb = load_framenet(framenet_dir)
            
    save_framenet_clusters(
        get_framenet_clusters(use_children=False),
        join(output_dir,"lus-wo-ch.csv"))

    save_framenet_clusters(
        get_framenet_clusters(use_children=False, verbs_only=True),
        join(output_dir, "lus-wo-ch-verbs.csv"))

    save_framenet_clusters(
        get_framenet_clusters(use_children=True),
        join(output_dir, "lus-with-ch-r.csv"))

    save_framenet_clusters(
        get_framenet_clusters(use_children=True, verbs_only=True),
        join(output_dir, "lus-with-ch-r-verbs.csv"))


def main():
    parser = argparse.ArgumentParser(description='Extracts verb clusters from the framenet.')
    parser.add_argument('framenet_dir', help='Directory with the framenet files.')
    parser.add_argument('output_dir', help='Output directory.')
    args = parser.parse_args()
    print("Input: ", args.framenet_dir)
    print("Output: ", args.output_dir)

    run(args.framenet_dir, args.output_dir)


if __name__ == '__main__':
    main()

