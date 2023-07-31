from __future__ import print_function
from pandas import read_csv
from traceback import format_exc
import codecs 


def build_graph(preframes_fpath, output_fpath):
    df = read_csv(preframes_fpath, sep="\t", encoding='utf8', names=["cluster","features"])

    with codecs.open(output_fpath, "w", "utf-8") as out:
        for i, row in df.iterrows():
            try:
                senses = [s.strip() for s in row.cluster.split(",")]
                if len(senses) <= 1: continue
                sense_0, _ = senses[0].split(":")

                for sense_sim_j in senses[1:]:
                    sense_j, sim_0j = sense_sim_j.split(":")
                    out.write("{}\t{}\t{}\n".format(sense_0, sense_j, int(sim_0j)))
                    out.write("{}\t{}\t{}".format(sense_j, sense_0, int(sim_0j)))    
            except:
                print(format_exc())
                
    print("Output graph:", output_fpath)


build_graph(
        preframes_fpath = "exact-full.txt", # e.g. tmp/verbs/exact-full.txt
        output_fpath = preframes_fpath + ".graph")

