#!/usr/bin/env python

from __future__ import print_function
import argparse
from pandas import read_csv 
from traceback import format_exc
from itertools import combinations, product
import codecs 


verbose = False 


def extract_verb_lus(lus_str):
    try:
        lus = set()
        for lu in lus_str.split(","):
            try:
                lu = lu.strip()
                if " " in lu or ".v" not in lu: continue
                lus.add(lu)
            except:
                if verbose: print(format_exc())
        return lus
    except:
        if verbose: print(lus_str)
        return set()


def extract_lus(lus_str):
    lus_bow = {}
    for lu in lus_str.split(","):
        try:
            lu_name, lu_freq = lu.split(":")
            lu_name = lu_name.strip()
            if " " in lu_name: continue
            lu_freq = int(lu_freq)
            lus_bow[lu_name] = lu_freq
        except:
            print(lu)
            
    return lus_bow


def count_lus(lus_str):
    total_freq = 0
    lus_bow = {}
    for lu in lus_str.split(","):
        try:
            lu_name, lu_freq = lu.split(":")
            lu_name = lu_name.strip()
            if " " in lu_name: continue
            lu_freq = int(lu_freq)
            lus_bow[lu_name] = lu_freq
            total_freq += lu_freq
        except:
            print(lu)
            
    return total_freq


def generate_triples(roles, frame2lus, min_lus_size = 10):
    top_roles = roles[roles.lus_size >= min_lus_size]

    with codecs.open(output_fpath, "w", "utf-8") as out:
        frame_num = 0
        frame_num_written = 0
        svo_triples_num = 0 

        for frame_name, role_groups in top_roles.groupby("frame"):
            frame_num += 1
            if len(role_groups) < 3: continue

            # Save two most frequent roles 
            frame_roles = {}
            for i, role in role_groups.iterrows():
                if role.role != "FEE" and len(frame_roles) < 2:
                    frame_roles[role.role] = role.lus_bow
            if len(frame_roles) != 2: continue
            label = u"\n# {}: {} FEE {}".format(frame_name,
                frame_roles.keys()[0], frame_roles.keys()[1])

            # Generate the triples from all roles plus the verb cluster
            fees = frame2lus.get(frame_name, set())
            if len(fees) == 0: continue

            out.write("{}\n".format(label))
            frame_num_written += 1
            for verb in fees:
                v = verb.replace(".v", "")
                for role1, role2 in combinations(frame_roles.keys(), 2):
                    role1_lus = frame_roles[role1]
                    role2_lus = frame_roles[role2]
                    for role1_lu, role2_lu in product(role1_lus, role2_lus):
                        out.write(u"{}\t{}\t{}\n".format(role1_lu, v, role2_lu))
                        svo_triples_num += 1

            #if frame_num > 10: break

    print("Number of frames:", frame_num)
    print("Number of written frames:", frame_num_written)
    print("Number of written triples:", svo_triples_num)
    print("Output:", output_fpath)


def run(roles_fpath, verbs_fpath, output_fpath, min_lus_size = 10):
    # Load verbs
    verbs = read_csv(verbs_fpath, sep="\t", encoding="utf-8")
    verbs["lus_bow"] = verbs.lus.apply(extract_verb_lus)
    frame2lus = {row.frame: row.lus_bow for i, row in verbs.iterrows() if len(row.lus_bow) > 0}
    
    # Load roles
    roles = read_csv(roles_fpath, sep="\t", encoding="utf-8", names=["frame", "role", "lus"])
    roles["lus_size"] = roles.lus.apply(count_lus)
    roles["lus_bow"] = roles.lus.apply(extract_lus)
    roles = roles.sort_values(["frame","lus_size"], ascending=False)
    
    # Generate triples from verbs and roles
    generate_triples(roles, frame2lus, min_lus_size)

    
def main():
    parser = argparse.ArgumentParser(description='Generates SVO triples from the framenet data.')
    parser.add_argument('roles_fpath', help='Path to a CSV file with the parsed framenet roles.')
    parser.add_argument('verbs_fpath', help='Path to a CSV file with the parsed framenet verb clusters.')
    parser.add_argument('output_fpath', help='Output file with the triples.')   
    parser.add_argument('min_lus_size', help='Minimum size of the LU of the roles.', type=int)
    args = parser.parse_args()
    print("Roles:", args.roles_fpath)
    print("Verbs:", args.verbs_fpath)
    print("Output:", args.output_fpath)
    print("Min LUs size:", args.min_lus_size)
    run (args.roles_fpath, args.verbs_fpath, args.output_fpath, args.min_lus_size)


if __name__ == '__main__':
    main()

# Example files on ltcpu3 
# roles_fpath = "/home/panchenko/verbs/fi/fi/eval/output/slots.csv"
# verbs_fpath = "/home/panchenko/verbs/fi/fi/eval/lus-without-children.tsv"
