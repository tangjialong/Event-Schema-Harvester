#!/usr/bin/env python

from __future__ import print_function
import argparse
from glob import glob
from collections import defaultdict
from collections import namedtuple
from os.path import join
import codecs 
import re
from traceback import format_exc
import operator
import codecs 
from collections import Counter
import networkx as nx
from networkx import NetworkXNoPath


test_sentence = """
# RID: 2492
# Frame "Claim_ownership"
#     FEE: 5
#     Property: 3
#     Claimant: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
# Frame "Leadership"
#     FEE: 18
#     Leader: 18
#     Governed: 19, 20
# Mapped annotation onto parse #1 with average recall of 0.792
1       Note    note    nn      NN      _       _       _       _       _       _       _       _       _
2       :       :       :       :       _       _       _       _       _       _       _       _       _
3       Taiwan  Taiwan  NP      NNP     _       _       _       5       nsubjpass       _       Property        _        _
4       is      be      VBZ     VBZ     _       _       _       5       auxpass _       _       _       _
5       claimed claim   VVN     VBN     _       _       _       1|1     ccomp|parataxis Claim_ownership _       _        _
6       by      by      in      IN      _       _       _       _       _       _       _       _       _
7       both    both    pdt     PDT     _       _       _       9       predet  _       _       _       _
8       the     the     dt      DT      _       _       _       9       det     _       _       _       _
9       Government      government      nn      NN      _       _       _       5       agent   _       Claimant _       _
10      of      of      in      IN      _       _       _       _       _       _       _       _       _
11      the     the     dt      DT      _       _       _       13      det     _       _       _       _
12      People's        People's        NP      NNP     _       _       _       13      nn      _       _       _        _
13      Republic        Republic        NP      NNP     _       _       _       9       prep_of _       _       _        _
14      of      of      in      IN      _       _       _       _       _       _       _       _       _
15      China   China   NP      NNP     _       _       _       13      prep_of _       _       _       _
16      and     and     cc      CC      _       _       _       _       _       _       _       _       _
17      the     the     dt      DT      _       _       _       18      det     _       _       _       _
18      authorities     authority       nns     NNS     _       _       _       9|13    prep_of|conj_and        _        _       Leadership      Leader
19      on      on      in      IN      _       _       _       _       _       _       _       _       _
20      Taiwan  Taiwan  NP      NNP     _       _       _       18      prep_on _       _       _       Governed
21      .       .       sent    .       _       _       _       _       _       _       _       _       _"""


sep_deps_list = "|"
no_parent = "_"
NO_PATH = ("","","")
verbose = False
frame_prefix = "# Frame "
role_prefix = "#     "
framename_prefix = '"'
role_key_value_sep = ":"


Frame = namedtuple('Frame', 'name roles')
Role = namedtuple('Role', 'name lu ids')
Dependency = namedtuple('Dependency', 'src token lemma pos dep dst')


def file_split(fpath, delim='\n\n', bufsize=1024):
    with codecs.open(fpath, "r", "utf-8") as f:
        prev = ''
        while True:
            s = f.read(bufsize)
            if not s:
                break
            split = s.split(delim)
            if len(split) > 1:
                yield prev + split[0]
                prev = split[-1]
                for x in split[1:-1]:
                    yield x
            else:
                prev += s
        if prev:
            yield prev
            

def parse_role(role_string, deps):
    """ A line with the row like: '#     FEE: 5'. """
    
    if not role_string.startswith(role_prefix):
        return None # not a role
    else:
        try:
            role_name, dst_ids = role_string[len(role_prefix):].split(role_key_value_sep)
            role_name = role_name.strip()
            
            dst_ids = [int(dst_id) for dst_id in dst_ids.split(",")]
            dst = " ".join([deps[dst_id].lemma for dst_id in dst_ids])
            return Role(role_name, dst, dst_ids)
        except:
            if verbose: print(format_exc())
            return None
   

def parse_sentence(sentence_str):
    """ Gets a CoNLL sentence with comments 
    and returns a list of frames. """
    
    deps = defaultdict()
    comment_str = ""
    for line in sentence_str.split("\n"):
        if line.startswith("# "):
            if len(line) > 1: comment_str += line + "\n"
        else:
            fields = line.split("\t")
            if len(fields) >= 10:
                dep = parse_dep(fields)
                deps[dep.src] = dep 
            else:
                if verbose: print("Bad line:", line)
                    
    frames = []
    for frame_str in comment_str.split(frame_prefix):
        frame_lines = frame_str.split("\n")
        if not frame_lines[0].startswith(framename_prefix):
            continue

        frame_name = frame_lines[0].replace('"','')
        del frame_lines[0]   
        frame = Frame(frame_name, list())

        for fl in frame_lines:
            if fl.startswith(role_prefix):
                role = parse_role(fl, deps)
                if role is not None:
                    frame.roles.append(role)
                else:
                    if verbose: print("Bad role:", fl)
        frames.append(frame)
        
    return frames, deps

    
def parse_dep(fields):
    if len(fields) < 10: 
        print("Bad dependency:", fields)
        return Dependency()
    
    return Dependency(
        src = int(fields[0]),
        token = fields[1],
        lemma = fields[2],
        pos = fields[3],
        dst = fields[8],
        dep = fields[9])


def extract_frames(conll_dir):
    conll_fpaths = glob(join(conll_dir,"*.conll"))
    name2frame = defaultdict(lambda: list())
    paths_fee2role = Counter() 
    paths_role2role = Counter() 
    
    sentences_total = 0
    frames_total = 0
    
    for file_num, conll_fpath in enumerate(conll_fpaths):
        if verbose: print("\n", "="*50, "\n", conll_fpath, "\n")

        for sent_num, sent_str in enumerate(file_split(conll_fpath)):
            frames, deps = parse_sentence(sent_str)
            
            for f in frames: name2frame[f.name].append(f)
            fee2role, role2role = get_syn_paths(frames, deps)
            paths_fee2role.update(fee2role)
            paths_role2role.update(role2role)
            
            frames_total += len(frames)
            sentences_total += 1 

    print("Sentences total:", sentences_total)
    print("Files total:", file_num + 1)
    print("Frames total:", frames_total)

    return name2frame, paths_fee2role, paths_role2role


def aggregate_frames(name2frame, output_fpath): 
    frame2role2lu = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    roles_count = 0
    for frame_name in name2frame:
        for frame in name2frame[frame_name]:
            for role in frame.roles:
                frame2role2lu[frame.name][role.name][role.lu] += 1 # update the count 
                roles_count += 1
    print("Roles count:", roles_count)
    
    save_frame2role2lu(frame2role2lu, output_fpath)


def save_frame2role2lu(frame2role2lu, output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as out:
        uniq_roles = 0
        for frame in frame2role2lu:
            for role in frame2role2lu[frame]:
                try:
                    lus = sorted(frame2role2lu[frame][role].items(), key=operator.itemgetter(1), reverse=True)
                    uniq_roles += 1
                    out.write(u"{}\t{}\t{}\n".format(
                        frame,
                        role,
                        u", ".join(u"{}:{}".format(
                            lu.replace(",", " ").replace(":"," "), count) for lu, count in lus)))
                except:
                    print("Bad entry:", frame, role, lus)
                    print(format_exc())
    print("Uniq. roles count:", uniq_roles)
    print("Output:", output_fpath)
 

def get_path(g, src, dst):
    """ Returns a list of labels between the src
    and the dst nodes. """
    
    try:
        path = nx.shortest_path(g, src, dst)
    except NetworkXNoPath:
        try:
            if verbose: print("Warning: trying inverse path.")
            path = nx.shortest_path(g, dst, src)
        except NetworkXNoPath:
            return NO_PATH
        
    labeled_path = []
    
    for i, src in enumerate(path):
        dst_index = i + 1
        if dst_index >= len(path): break
        dst = path[dst_index]
        labeled_path.append((
            g.nodes[src]["label"],
            g[src][dst]["label"],
            g.nodes[dst]["label"]))

    words = "-->".join([s for s, t, d in labeled_path])
    deps = "-->".join([t for s, t, d in labeled_path])
    deps_n_words = ":".join(["%s--%s--%s" % (s, t,d ) for s, t, d in labeled_path])
    
    return words, deps, deps_n_words


def build_nx_graph(list_of_nodes, list_of_edges):
    """ Takes as an input a list of triples (src, dst, label)
    and builds the graph out of them. """
    
    g = nx.DiGraph()
    
    for src, label in list_of_nodes:
        g.add_node(src, label=label)
    
    for src, dst, label in list_of_edges:
        g.add_edge(src, dst, label=label)
        
    return g


def build_graph(deps):
    nodes = []
    edges = []
    
    for d in deps.values():
        if d.dst == no_parent or d.dep == no_parent:
            nodes.append( (d.src, d.lemma) )
        else:
            dst_ids = [int(dst_id) for dst_id in d.dst.split(sep_deps_list)] 
            dst_types = d.dep.split(sep_deps_list)
            dsts = zip(dst_ids, dst_types)
            nodes.append( (d.src, d.lemma) ) 
            for dst_id, dst_label in dsts:
                edges.append( (d.src, dst_id, dst_label) ) 
              
    return build_nx_graph(nodes, edges)


def get_syn_paths(frames, deps):
    """ Returns shortest paths between FEE and roles and 
    roles in the parse trees. """
    
    fee2role = Counter()
    role2role = Counter()
    
    for frame in frames:
        fee_ids = []
        role_ids_list = []
        for role in frame.roles:
            if role.name == "FEE":
                fee_ids = role.ids
            else:
                role_ids_list.append(role.ids)
        
        g = build_graph(deps)
        
        print("="*50, "\n", "fee2role","\n")
        for fee_id in fee_ids:
            for role_id_list in role_ids_list:
                for role_id in role_id_list:
                    if fee_id != role_id:
                        path = get_path(g, fee_id, role_id)
                        if path != NO_PATH:
                            print("\t".join(path), "\n")
                            fee2role.update({path[1]:1})
        
        print("\n","="*50, "\n", "role2role","\n")
        for role_id_list_i in role_ids_list:
            for role_id_list_j in role_ids_list:
                if role_id_list_i == role_id_list_j: continue
                for role_id_i in role_id_list_i:
                    for role_id_j in role_id_list_j:
                        if role_id_i == role_id_j: continue
                        path = get_path(g, role_id_i, role_id_j)
                        if path != NO_PATH:
                            print("\t".join(path), "\n")
                            role2role.update({path[1]:1})
        
    return fee2role, role2role


def save_path_stats(x2role, output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as out:
        for path, freq in sorted(x2role.items(), key=operator.itemgetter(1), reverse=True):
            if freq > 2:
                out.write("{}\t{}\n".format(path, freq))

    print("Output:", output_fpath)


def run(conll_dir, output_dir):
    fee2role_fpath = join(output_dir, "fee2role.csv")
    role2role_fpath = join(output_dir, "role2role.csv")
    slots_fpath = join(output_dir, "slots.csv")
    
    name2frame, fee2role, role2role = extract_frames(conll_dir)
    save_path_stats(fee2role, fee2role_fpath)
    save_path_stats(role2role, role2role_fpath)
    aggregate_frames(name2frame, slots_fpath)
    

def main():
    parser = argparse.ArgumentParser(description='Extracts an evaluation dataset for role induction'
            'from the conll framenet files.')
    parser.add_argument('conll_dir', help='Directory with the .conll files (dep.parsed framenet).')
    parser.add_argument('output_dir', help='Output directory.')
    args = parser.parse_args()
    print("Input: ", args.conll_dir)
    print("Output: ", args.output_dir)

    run(args.conll_dir, args.output_dir)


if __name__ == '__main__':
    main()

