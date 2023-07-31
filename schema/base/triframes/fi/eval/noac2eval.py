from glob import glob
from os.path import join
from pandas import read_csv
import codecs 
import argparse


sample_input = """
Density Variance    Average Coverage    Objects coverage    Attributes coverage Conditions coverage Extent         Intent   Modus
31.23   7899693.28  1286.43 0.48    0.64    0.33    0.14    <take#VB, see#VB, think#VB, make#VB, have#VB, know#VB, say#VB, tell#VB, do#VB, get#VB>  <person, man, player, fan, nobody, anyone, lot, be, everyone, friend, many, guy,  everybody, reader, God, other, child, people, someone, one, eye, kid> <picture, anything, video, movie, something, thing, lot>
"""


sample_output = """

# Cluster 23

Predicates: dred, donna, hank, ggg, kardashian, haff, jill, santa, julie, luke, bday, lizzie, beatle, publi, eather, sasha, lauren, bernie, kern, 0ne, lfe, monica, kristen, hanna, sherman, suckin, kate, hud, zack, kirsten, einstein, ^_^
Subjects: hun, hoo
Objects: tape, Tape

# Cluster 23

Predicates: dred, donna, hank, ggg, kardashian, haff, jill, santa, julie, luke, bday, lizzie, beatle, publi, eather, sasha, lauren, bernie, kern, 0ne, lfe, monica, kristen, hanna, sherman, suckin, kate, hud, zack, kirsten, einstein, ^_^
Subjects: hun, hoo
Objects: tape, Tape
"""


def delete_header(input_fpath, output_fpath):
    with codecs.open(input_fpath, 'r', "utf-16le") as fin, codecs.open(output_fpath, 'w', "utf-8") as fout:
        data = fin.read().splitlines(True)[1:]
        fout.writelines(data)

        
def exract_bow(df, old_field, new_field):
    df[new_field] = df[old_field].apply(lambda x: x
                                       .replace("<","")
                                       .replace(">","")
                                       .split(","))

    df[new_field] = df[new_field].apply(lambda p_set: set([p.split("#")[0] for p in p_set]) )
    df[new_field] = df[new_field].apply(lambda p_set: [p.strip() for p in p_set if p.strip() != ""] )
    
    return df


def run(noac_dir):
    for noac_fpath in glob(join(noac_dir,"*.txt")):
        csv_fpath = noac_fpath.replace(" ", "").replace(",","-") + ".tsv" 
        delete_header(noac_fpath, csv_fpath)

        df = read_csv(csv_fpath, sep="\t", encoding="utf8")
        df = exract_bow(df, "Extent", "predicates")
        df = exract_bow(df, "Intent", "subjects")
        df = exract_bow(df, "Modus", "objects")

        output_fpath = csv_fpath + ".arguments.tsv"
        with codecs.open(output_fpath, "w", "utf-8") as out:
            for i, row in df.iterrows():
                out.write("# Cluster {:d}\n\n".format(i))
                out.write("Predicates: {}\n".format(", ".join(row.predicates)))
                out.write("Subjects: {}\n".format(", ".join(row.subjects)))
                out.write("Objects: {}\n\n".format(", ".join(row.objects)))

        print("Output:", output_fpath)


def main():
    parser = argparse.ArgumentParser(description='Converts NOAC outputs to the input of the evaluation script. ')
    parser.add_argument('noac_dir', help='A directory with the NOAC output files.')  
    args = parser.parse_args()
    print("Input:", args.noac_dir)
    run(args.noac_dir)


if __name__ == '__main__':
    main()

# Example paths of ltcpu3
# noac_dir = "/home/panchenko/tmp/triclustering-results/"
