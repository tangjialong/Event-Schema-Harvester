from __future__ import print_function
import argparse
import gzip 
from os.path import join


verbose = False


def load_framenet_svo(framenet_svo_fpath):
    """ Load the framenet svo: SVO + comment lines """
    
    fn_svo = set()
    with gzip.open(framenet_svo_fpath, "rt", encoding="utf-8") as fn:
        for i, line in enumerate(fn):
            try:
                if i % 1000000 == 0: print(i / 1000000, "million of svo triples")
                if line.startswith("# ") or len(line) < 5: continue
                s, v, o = line.split("\t")
                o = o.strip()
                fn_svo.add( (s,v,o) )
            except KeyboardInterrupt:
                break
            except:
                if verbose: print("Bad line:", line)

    return fn_svo

            
def extract_lemma(lemma_pos):
    lemma, pos = lemma_pos.split("#")
    lemma = lemma.lower()
    return lemma


def load_depcc_svo(depcc_svo_fpath):
    """Load the depcc svo in the format 'help#VB you#PR_I#PR 72040.0' """ 

    dep_svo = set()
    with gzip.open(depcc_svo_fpath, "rt", encoding="utf-8") as dep:
        for i, line in enumerate(dep):
            try:
                line = line.replace("_", "\t")
                if i % 1000000 == 0: print(i / 1000000, "million of svo triples")
                v_pos, s_pos, o_pos, freq = line.split("\t")
                v = extract_lemma(v_pos)
                s = extract_lemma(s_pos)
                o = extract_lemma(o_pos)
                dep_svo.add( (s,v,o) )
            except KeyboardInterrupt:
                break
            except:
                if verbose: print("Bad line:", line)

    return dep_svo


def filter_depcc_svo(depcc_svo_fpath, output_fpath, common_svo):
    """Filter the depcc svo in the format 'help#VB you#PR_I#PR 72040.0'
    according to the common triples with the framenet. """ 

    with gzip.open(depcc_svo_fpath, "rt", encoding="utf-8") as dep_in, gzip.open(output_fpath, "wt", encoding="utf-8") as dep_out:
        svo_num = 0
        common_svo_num = 0
        for i, line in enumerate(dep_in):
            try:
                line = line.replace("_", "\t")
                if i % 1000000 == 0: print(i / 1000000, "million of svo triples")
                v_pos, s_pos, o_pos, freq = line.split("\t")
                v = extract_lemma(v_pos)
                s = extract_lemma(s_pos)
                o = extract_lemma(o_pos)
                svo_num += 1
                if (s,v,o) in common_svo:
                    dep_out.write(line)
                    common_svo_num += 1
            except KeyboardInterrupt:
                break
            except:
                if verbose: print("Bad line:", line)
    
    print("Number of frames svo triples:", svo_num)
    print("Number of common frame + dep svo triples:", common_svo_num)
    print("Output:", output_fpath)

    
def filter_framenet_svo(framenet_svo_fpath, output_fpath, dep_svo):
    """ Filter the framenet svo provided in the form of 'svo + comment lines'
    according the the list of dependency parsed svo triples. """
    
    with gzip.open(framenet_svo_fpath, "rt", encoding="utf-8") as fn_in, gzip.open(output_fpath, "wt", encoding="utf-8") as fn_out:
        svo_num = 0
        common_svo_num = 0
        for i, line in enumerate(fn_in):
            try:
                if i % 1000000 == 0: print(i / 1000000, "million of svo triples")
                if line.startswith("# ") or len(line) < 5: fn_out.write(line)
                
                s, v, o = line.split("\t")
                o = o.strip()
                svo_num += 1
                if (s,v,o) in dep_svo:
                    fn_out.write(line)
                    common_svo_num += 1
            except KeyboardInterrupt:
                break
            except:
                if verbose: print("Bad line:", line)

    print("Number of frames svo triples:", svo_num)
    print("Number of common frame + dep svo triples:", common_svo_num)
    print("Output:", output_fpath)

                    
def run(framenet_svo_fpath, depcc_svo_fpath, output_dir):
    # Load triples
    fn_svo = load_framenet_svo(framenet_svo_fpath)
    dep_svo = load_depcc_svo(depcc_svo_fpath)

    # Make the intersection
    common_svo = fn_svo.intersection(dep_svo)
    print("Number of common svo triples:", len(common_svo))
    
    # Save the intersection 
    output_fn_fpath = join(output_dir, "framenet-common-triples.tsv.gz")
    filter_framenet_svo(framenet_svo_fpath, output_fn_fpath, common_svo)
    
    output_depcc_fpath = join(output_dir, "depcc-common-triples.tsv.gz")    
    filter_depcc_svo(depcc_svo_fpath, output_depcc_fpath, common_svo)

def main():
    parser = argparse.ArgumentParser(description='Extracts common SVO triples: framenet + depcc. ')
    parser.add_argument('framenet_svo_fpath', help='Path to the framenet SVO triples.')  
    parser.add_argument('depcc_svo_fpath', help='Path the the depcc SVO triples.')
    parser.add_argument('output_dir', help='Output directory.')
    args = parser.parse_args()
    print("Framenet SVOs:", args.framenet_svo_fpath)
    print("DepCC SVOs:", args.depcc_svo_fpath)
    print("Output:", args.output_dir)

    run(args.framenet_svo_fpath, args.depcc_svo_fpath, args.output_dir)


if __name__ == '__main__':
    main()

# Example paths of ltcpu3
# framenet_svo_fpath = "/home/panchenko/verbs/fi/fi/eval/framenet-triples-full-lus1.tsv.gz"
# depcc_svo_fpath = "/home/panchenko/tmp/verbs/frames/2.6t/vso-26m.csv.gz"


