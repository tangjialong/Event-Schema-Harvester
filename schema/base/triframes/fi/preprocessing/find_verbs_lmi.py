from __future__ import print_function
import gzip 
from time import time
import codecs
from os.path import join
from traceback import format_exc
import re


def find_verbs(lmi_fpath, voc=[]):
    use_voc = "" if len(voc) == 0 else ".voc"
    output_fpath = lmi_fpath + use_voc + ".verbs.csv"
    wfc_num = 0
    verbs_num = 0
    tic = time()
    re_verb = re.compile(r"^[a-zA-Z]+#VB( +[a-zA-Z]+#(VB|RP|RB|IN))?$")

    with gzip.open(lmi_fpath, "r") as lmi, codecs.open(output_fpath, "w") as out:
        for line in lmi:
            try:
                wfc_num += 1
                word, feature, score = line.split("\t")
                word = word.strip()
                if len(voc) > 0 and (" " in word or word.split("#")[0] not in voc):
                    continue
                if not re_verb.match(word):
                    continue
                out.write(line)
                verbs_num += 1
            except:
                print("Warning: bad line '{}'", line)
                print(format_exc())

    print("Time: {} sec.".format(time() - tic))
    print("Found verbs:", verbs_num, "of", wfc_num)
    print("Input:", lmi_fpath)
    print("Output:", output_fpath)
    
    return output_fpath

lmi_fpath = "lmi-culwg-coarse.csv.gz"
voc = ["run", "jog", "train", "execute", "launch", "eat", "consume", "chew"]

find_verbs(lmi_fpath, voc=voc)
find_verbs(lmi_fpath)
