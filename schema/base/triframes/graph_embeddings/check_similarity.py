import pickle
import sys
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors

embeddings_name = "input100_connection2_dim200_windowsize70.emb"
id_to_check = "27236"

with open('id_to_name.pkl', 'rb') as f:
    pkl = pickle.load(f)

vectors = KeyedVectors.load_word2vec_format(embeddings_name, binary=False)

node_to_check = pkl[id_to_check]
similar = wv_from_text.most_similar(positive=id_to_check)

print("similar to: ", node_to_check["verb"], node_to_check["subject"], node_to_check["object"])
print("--------------")

for index, score in similar:
    node = pkl[index]
    print(node["verb"], node["subject"], node["object"], score)