import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nxviz.plots import CircosPlot
import numpy as np
import itertools
import pickle

lines_to_read = 100000

def get_pairs(df, col1, col2):
    sub = df[df.duplicated(subset=[col1, col2], keep=False)]
    grp_by = sub.groupby([col1, col2])
    pairs = []
    for i, group in enumerate(grp_by.groups):
        try:
            grp = grp_by.get_group(group)
            if len(grp) > 1:
                pairs += list(itertools.combinations(grp.index.values, 2))
        except KeyError:
            print("KeyError")
    return pairs

#read data
df = pd.read_csv("vso-1.3m-pruned-strict.csv", delimiter="\t", header=None,  nrows=lines_to_read)
df.columns = ['verb', 'subject', 'object', 'score']
df = df.reset_index()
df = df.fillna('')

#init graph
G=nx.Graph()
edges = []

dictionary_id_to_name = {}

#add vertices
print("Adding vertices...")
for index, row in df.iterrows():
    G.add_node(index, verb=row['verb'], subject=row['subject'], object=row['object'])
    dictionary_id_to_name[str(index)] = row
print("Done")

#add edges
edges += get_pairs(df, 'verb','subject')
edges += get_pairs(df, 'verb','object')
edges += get_pairs(df, 'object','subject')

G.add_edges_from(edges)

# graph info
print ("nodes: ", G.number_of_nodes())
print ("edges: ", G.number_of_edges())

#save graph
nx.write_adjlist(G, "triframes.adjlist")
nx.write_edgelist(G, "triframes.edgelist")

#save dictionary with id mapping
with open('id_to_name.pkl', 'wb') as f:
    pickle.dump(dictionary_id_to_name, f, pickle.HIGHEST_PROTOCOL)

#plot graph
#c = CircosPlot(G, node_labels=True)
#c.draw()
#plt.show()


