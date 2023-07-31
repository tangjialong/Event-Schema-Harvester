from __future__ import print_function
import gzip
from fi.exact_full.exact_full import skip, generalize_by_pos
import codecs
from traceback import format_exc


def prune_word_features(svo_fpath, strict=True, generalize=True, verbose=False):
    output_fpath = svo_fpath + "-pruned-strict{}.csv".format(int(strict))

    with gzip.open(svo_fpath, "rb") as svo_bytes, codecs.open(output_fpath, "w", "utf-8") as out:
        reader = codecs.getreader("utf-8")
        svo = reader(svo_bytes)

        for i, line in enumerate(svo):
            try:
                verb, subj_obj, freq = line.split("\t")
                subj, obj = subj_obj.split("_")
                subj_lemma, subj_pos = subj.split("#")
                obj_lemma, obj_pos = obj.split("#")
                freq = float(freq)

                if not skip(subj_lemma, subj_pos, obj_lemma, obj_pos, not strict):
                    out.write(u"{}\t{}\t{}\t{}\n".format(
                        verb,
                        generalize_by_pos(subj_lemma, subj_pos),
                        generalize_by_pos(obj_lemma, obj_pos),
                        freq))
                else:
                    if verbose: print("Skipping:", line.strip())
            except:
                if verbose:
                    print(line.strip())
                    print(format_exc())
        print("Output:", output_fpath)

        
svo_fpath = "wf.csv.gz"  # e.g. "ltcpu3:/home/panchenko/tmp/verbs/frames/2.6t/wf.csv.gz"
prune_word_features(svo_fpath, strict=True, generalize=True, verbose=False)
prune_word_features(svo_fpath, strict=False, generalize=True, verbose=False)
