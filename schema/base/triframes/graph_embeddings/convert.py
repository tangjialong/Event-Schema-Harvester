import pickle
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace numerical ids with triples')
    parser.add_argument('--inputfile', default='input.emb', help='Input embedding')
    parser.add_argument('--outputfile', default='output.emb', help='Output embedding')
    parser.add_argument('--dict', default='id_to_name.pkl', help='pkl file with dictionary')
    args = parser.parse_args()

with open(args.dict, 'rb') as f:
    pkl = pickle.load(f)

with open(args.inputfile) as f, open(args.outputfile, "w") as f1:
    #copy first line with dimensions
    f1.write(f.readline())
    for line in f:
        node_index = line.split(" ")[0]
        triple = pkl.get(node_index)
        output_str = "(" + triple['verb'] + " " + triple['subject'] + " " + triple['object'] + ")"
        f1.write(output_str + line[len(node_index):])

