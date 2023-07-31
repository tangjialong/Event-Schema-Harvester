import os
import rbo
import json
import math
import argparse
import numpy as np
import pickle as pk
from tqdm import tqdm
import prettytable as pt
from collections import defaultdict
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from transformers import BertForMaskedLM, BertTokenizer

from base.baseline_kmeans import kmeans
from base.baseline_agglo import agglo
from base.baseline_jcsc import jcsc
from base.baseline_triframes import triframes
from base.baseline_etyeclus import etyeclus
from harvester import esher

def prepare_sentence(token_list, tokenizer):
    '''
    Inputs:
        token_list: a list of tokens
        tokenizer: a TokenizerClass object from HuggingFace
    Return:
        tokenized_text: A list of tokens obtained from basic tokenizer (e.g., WhiteSpaceTokenizer)
        tokenized_to_id_indicies: A list of (tokenids_chunks_index, token_id_start_index, token_id_end_index)
        tokenids_chunks: A list of token_id_end_index
    '''
    # setting for BERT
    model_max_tokens = 512
    has_sos_eos = True
    ############## ########
    max_tokens = model_max_tokens
    if has_sos_eos:
        max_tokens -= 2
    sliding_window_size = max_tokens // 2

    if not hasattr(prepare_sentence, 'sos_id'):
        prepare_sentence.sos_id, prepare_sentence.eos_id = tokenizer.encode('', add_special_tokens=True)

    # tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
    tokenized_text = token_list
    tokenized_to_id_indicies = []

    tokenids_chunks = []  # useful only if the sentence is longer than max_tokens
    tokenids_chunk = []

    for index, token in enumerate(tokenized_text + [None]):
        if token is not None:
            tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
            tokenids_chunks.append([prepare_sentence.sos_id] + tokenids_chunk + [prepare_sentence.eos_id])
            if sliding_window_size > 0:
                tokenids_chunk = tokenids_chunk[-sliding_window_size:]
            else:
                tokenids_chunk = []
        if token is not None:
            tokenized_to_id_indicies.append((len(tokenids_chunks),
                                             len(tokenids_chunk),
                                             len(tokenids_chunk) + len(tokens)))
            tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

    return tokenized_text, tokenized_to_id_indicies, tokenids_chunks

def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()

def sentence_encode(tokens_id, model):
    input_ids = torch.tensor([tokens_id], device=model.device)

    with torch.no_grad():
        last_hidden_states = model(input_ids).last_hidden_state
        
    layer_embedding = tensor_to_numpy(last_hidden_states.squeeze(0))[1: -1]
    return layer_embedding

def sentence_to_wordtoken_embeddings(layer_embeddings, tokenized_text, tokenized_to_id_indicies):
    word_embeddings = []
    for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
        word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
    assert len(word_embeddings) == len(tokenized_text)
    return np.array(word_embeddings)

def handle_sentence(model, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
    layer_embeddings = [
        sentence_encode(tokenids_chunk, model) for tokenids_chunk in tokenids_chunks
    ]
    word_embeddings = sentence_to_wordtoken_embeddings(layer_embeddings,
                                                       tokenized_text,
                                                       tokenized_to_id_indicies)
    return word_embeddings

def process_sentence(token_list, tokenizer, model):
    tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(token_list, tokenizer)
    contextualized_word_representations = handle_sentence(model, tokenized_text, tokenized_to_id_indicies, tokenids_chunks)
    return contextualized_word_representations

def predict_masked_words(sentence_w_mask, model, tokenizer, top_k=50):
    '''
    sentence_w_mask: `str`, a single sentence with [MASK] token
    model: a BertForMaskedLM model
    '''
    inputs = tokenizer(sentence_w_mask, return_tensors='pt').to(model.device)

    masked_position_indice = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0], axis=1)

    word_prob, word_index = torch.topk(probs[masked_position_indice], top_k, dim=-1)
    word_prob = tensor_to_numpy(word_prob)
    word_index = tensor_to_numpy(word_index)

    predicted_words = []
    for i in range(len(word_prob)):    
        pred_word = []
        for r, j in enumerate(word_index[i]):
            pred_word.append([tokenizer.ids_to_tokens[j], float(word_prob[i][r])])
        predicted_words.append(pred_word)
    return predicted_words

####################################################################################################

lemmatizer = WordNetLemmatizer()

def aggregate_sense_feature_by_ranking(bert_expanded_verbnet):
    ''' Return lemma -> sense with features
    {
        lemma 1: {
            sense1: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
            sense2: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
        },
        lemma 2: {
            sense1: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
            sense2: [{'expd_lemma1': score, 'expd_lemma2': score, ...}, sense_embed],
        }
    }
    '''
    lemma2sense_features = {}
    for lemma, info in tqdm(bert_expanded_verbnet.items(), desc='generate verb sense features'):
        lemma2sense_features[lemma] = {}
        for sense_id, sense in enumerate(info['senses']):
            sense_feature = defaultdict(float)
            sense_embed = sense['sense_embed']
            for example_w_expansion in sense['examples_w_expansion']:
                for ele in example_w_expansion[1]:
                    expd_results = ele['expansion_results']
                    expd_lemma2rank = {}
                    for expd_lemma_rank, term in enumerate(expd_results):
                        expd_lemma = lemmatizer.lemmatize(term['token_str'], 'v')
                        if expd_lemma not in expd_lemma2rank:
                            expd_lemma2rank[expd_lemma] = 1+expd_lemma_rank
                        
                    for expd_lemma, expd_lemma_rank in expd_lemma2rank.items():
                        sense_feature[expd_lemma] += 1.0/math.log(1+expd_lemma_rank)
            sense_feature = dict(sense_feature)
            sorted_features = {ele[0]:ele[1] for ele in sorted(sense_feature.items(), key=lambda x:-x[1])}
            lemma2sense_features[lemma][sense_id] = [sorted_features, sense_embed]
    return lemma2sense_features

def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)

def disambiguate_word_sense(verb_info, lemma2sense_features, options):
    '''
    verb_info corresponds to one verb mention and is represented as a dict {
        'verb': str,
        'verb_embed': str,
        'verb_expansion_results': List[{'token_str': str, 'score': float}]
    }
    options is a dictionary of potential wsd parameters
    '''
    sense_top_k = options.get('sense_top_k', -1)

    lemma = verb_info['verb']
    expd_results = verb_info['verb_expansion_results']
    lemma_embed = verb_info['verb_embed']
    if lemma_embed is None:  # not salient predicate mention thus no embedding feature
        return -2
    if lemma not in lemma2sense_features:  # lemma does not appear in the dictionary
        return -1
    if len(lemma2sense_features[lemma]) == 1:  # only one sense, nothing to disambiguate
        return 0
    
    # get verb mention features
    expd_lemma2score = defaultdict(float)
    for rank, ele in enumerate(expd_results):
        expd_lemma = lemmatizer.lemmatize(ele[0], 'v')
        expd_lemma2score[expd_lemma] += (1.0/math.log(2+rank))
    expd_lemma2score = dict(expd_lemma2score)
    sorted_expd_lemma = [ele[0] for ele in sorted(expd_lemma2score.items(), key=lambda x:-x[1])]
    
    # start disambiguation
    sense_id2rbo_score = {}
    sense_id2embed_score = {}
    for sense_id, sense_feature in lemma2sense_features[lemma].items():
        ranked_sense_feature = list(sense_feature[0].keys())[:sense_top_k]
        rbo_score = rbo.RankingSimilarity(sorted_expd_lemma, ranked_sense_feature).rbo()
        sense_id2rbo_score[sense_id] = rbo_score

        sense_embed = sense_feature[1]
        if sense_embed is None or lemma_embed is None:
            embed_score = 0.0
        else:
            embed_score = cosine_similarity_embedding(sense_embed, lemma_embed)
        sense_id2embed_score[sense_id] = embed_score

    sense_id2final_score = {
        sense_id: sense_id2rbo_score[sense_id]*embed_score 
        for sense_id, embed_score in sense_id2embed_score.items()
    }
    sorted_senses = sorted(sense_id2final_score.items(), key=lambda x:-x[1])
    return sorted_senses[0][0]

####################################################################################################

def tfidf(item_list, item2features, item2freq):
    item_doc = []
    for item in item_list:
        features = item2features[item]
        doc = ''
        for k, v in features.items():
            cnt = math.ceil(v * item2freq[item])
            doc += ((k+' ')*cnt)
        item_doc.append(doc)
    
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X = vectorizer.fit_transform(item_doc)
    return X

def pca(X, pca_dim=50):
    pca_dim = min(pca_dim, X.shape[1]-1)
    svd = TruncatedSVD(pca_dim)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    print('Explained variance of the SVD step: {}%'.format(int(explained_variance * 100)))
    return X
    
def tfidf_pca(item_list, item2features, item2freq, pca_dim=50):
    X_init = tfidf(item_list, item2features, item2freq)
    X = pca(X_init, pca_dim=pca_dim)
    return X

####################################################################################################

def prepare_for_baselines(gpu_id, data_dir, corpus, salient_verbs, salient_obj_heads, obj2obj_head_infos, verb_sense_dict, original_corpus):
    print ('Loading Transformer model...')
    model_class, tokenizer_class, pretrained_weights = BertForMaskedLM, BertTokenizer, 'resources/bert-large-uncased-whole-word-masking'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    mlm_model = model_class.from_pretrained(pretrained_weights)
    mlm_model.eval()
    mlm_model = mlm_model.to(torch.device(f'cuda:{gpu_id}'))
    # get model for embedding extraction
    model = mlm_model.bert

    print ('Start feature extraction... ')
    svo_id2features = {}
    svo_id2features_without_emb = {}
    for sent_id in tqdm(corpus):
        sent_info = corpus[sent_id]
        if len(sent_info['svos']) == 0:
            continue

        # collect all svo_ids in this sentence that needs to be processed
        to_processed_svo_id_list = []
        for svo_id, svo in enumerate(sent_info['svos']):
            verb_lemma = svo[1][0]
            if verb_lemma.startswith("!"):
                verb_lemma = verb_lemma[1:]
            if verb_lemma in salient_verbs:
                if svo[2] is not None:
                    obj = svo[2][0]
                    obj_head_info = obj2obj_head_infos.get(obj, {})
                    if len(obj_head_info) != 0:
                        obj_head_lemma = obj_head_info['obj_head_lemma']
                        if obj_head_lemma in salient_obj_heads:
                            to_processed_svo_id_list.append(svo_id)

        if len(to_processed_svo_id_list) > 0:
            token_list = sent_info['token_list']
            token_list = [token.lower() for token in token_list]
            token_embeds = process_sentence(token_list, tokenizer, model)
            for svo_id in to_processed_svo_id_list:
                svo_index = f'{sent_id}_{svo_id}'
                svo = sent_info['svos'][svo_id]
                verb_index = svo[1][1]
                obj = svo[2]
                obj_range = svo[2][1]

                verb_match_flag = False
                obj_match_flag = False
                event_type = None
                # should be consisted with create_corpus.py
                instance = json.loads(original_corpus[int(sent_id)])
                for event in instance['event_mentions']:
                    # select event mentions whose trigger is a single token verb (refer to Shen et.al., EMNLP2021, Corpus-based Open-Domain Event Type Induction)
                    if event['trigger']['end'] - event['trigger']['start'] == 1 and verb_index == event['trigger']['start']:
                        verb_match_flag = True
                        event_type = event['event_type']
                    for argument in event['arguments']:
                        for entity in instance['entity_mentions']:
                            # non-pronoun argument that has some overlap with the object of our extracted (refer to Shen et.al., EMNLP2021, Corpus-based Open-Domain Event Type Induction)
                            if entity['id'] == argument['entity_id'] and entity['mention_type'] != 'PRO' and ((obj_range[0] >= entity['start'] and obj_range[0] < entity['end']) or (obj_range[-1] >= entity['start'] and obj_range[-1] < entity['end'])):
                                obj_match_flag = True

                if verb_match_flag and obj_match_flag:
                    verb_lemma = svo[1][0]
                    if verb_lemma.startswith('!'):  
                        verb_lemma = verb_lemma[1:]
                    verb_embed = token_embeds[verb_index]
                    verb_masked_sent = ' '.join(token_list[:verb_index] + ['[MASK]'] + token_list[verb_index+1:])
                    verb_expand_res = predict_masked_words(verb_masked_sent, mlm_model, tokenizer)[0]

                    obj = svo[2][0]
                    obj_head_info = obj2obj_head_infos.get(obj, {})
                    obj_head = obj_head_info["obj_head_lemma"]
                    obj_head_index = obj_range[obj_head_info['obj_head_relative_index']]
                    obj_head_embed = token_embeds[obj_head_index]
                    obj_head_masked_sent = ' '.join(token_list[:obj_head_index] + ['[MASK]'] + token_list[obj_head_index+1:])
                    obj_head_expand_res = predict_masked_words(obj_head_masked_sent, mlm_model, tokenizer)[0]

                    svo_id2features[svo_index] = {
                        'verb': verb_lemma,
                        'verb_index': verb_index,
                        'verb_embed': verb_embed,
                        'verb_expansion_results': verb_expand_res,
                        'obj': obj,
                        'obj_head': obj_head,
                        'obj_head_index': obj_head_index,
                        'obj_head_embed': obj_head_embed,
                        'obj_head_expansion_results': obj_head_expand_res,
                        'event_type': event_type,
                    }

                    svo_id2features_without_emb[svo_index] = {
                        'verb': verb_lemma,
                        'verb_index': verb_index,
                        'obj': obj,
                        'obj_head': obj_head,
                        'obj_head_index': obj_head_index,
                        'event_type': event_type,
                    }

    print (f'Select {len(svo_id2features)} mentions')
    with open(os.path.join(data_dir, 'corpus_parsed_svo_po_mention_features_for_eval.pk'), 'wb') as f:
        pk.dump(svo_id2features, f)
    with open(os.path.join(data_dir, 'corpus_parsed_svo_po_mention_features_for_eval.json'), 'w') as f:
        json.dump(svo_id2features_without_emb, f)

    ####################################################################################################

    parsed_data = pk.load(open(os.path.join(data_dir, 'corpus_parsed_svo_po_mention_features_for_eval.pk'), 'rb'))

    lemma2sense_features = aggregate_sense_feature_by_ranking(verb_sense_dict)
    parsed_data_wsd = {}
    options = {}
    options['sense_top_k'] = 10
    for svo_id, svo in tqdm(parsed_data.items(), desc='disambiguate svo verbs'):
        sense_id = disambiguate_word_sense(svo, lemma2sense_features, options)
        parsed_data_wsd[svo_id] = sense_id
    
    with open(os.path.join(data_dir, 'corpus_parsed_svo_po_mention_disambiguated_for_eval.json'), 'w') as f:
        json.dump(parsed_data_wsd, f)

    ####################################################################################################

    po_mentions = pk.load(open(os.path.join(data_dir, 'corpus_parsed_svo_po_mention_features_for_eval.pk'), 'rb'))
    svo_id2sense_id = json.load(open(os.path.join(data_dir, 'corpus_parsed_svo_po_mention_disambiguated_for_eval.json'), 'r'))

    print ('Collecting P-O mention features...')
    sense2embed_list = defaultdict(list)
    sense2expd_lemma_w_weight = {}
    sense2freq = defaultdict(int)
    obj_head2embed_list = defaultdict(list)
    obj_head2expd_lemma_w_weight = {}
    obj_head2freq = defaultdict(int)
    processed_svo_cnt = 0
    for svo_id, svo_info in tqdm(po_mentions.items()):
        verb = svo_info['verb']
        verb_embed = svo_info['verb_embed']
        verb_expd_results = svo_info['verb_expansion_results']
        obj_head = svo_info['obj_head']
        obj_head_embed = svo_info['obj_head_embed']
        obj_head_expd_results = svo_info['obj_head_expansion_results']

        # process verb     
        expd_lemma2weight = defaultdict(float)
        for ele in verb_expd_results:
            expd_lemma = lemmatizer.lemmatize(ele[0], 'v')
            expd_lemma2weight[expd_lemma] += ele[1]
        expd_lemma2weight = dict(expd_lemma2weight)

        sense_id = svo_id2sense_id[svo_id]
        sense = f'{verb}_{sense_id}'
        sense2embed_list[sense].append(verb_embed)
        sense2freq[sense] += 1
        if sense not in sense2expd_lemma_w_weight:
            sense2expd_lemma_w_weight[sense] = defaultdict(float)

            for expd_lemma, weight in expd_lemma2weight.items():
                sense2expd_lemma_w_weight[sense][expd_lemma] += weight
                
        # process obj (head)
        if obj_head_embed is not None:  # not salient object head
            expd_lemma2weight = defaultdict(float)
            for ele in obj_head_expd_results:
                expd_lemma = lemmatizer.lemmatize(ele[0])
                expd_lemma2weight[expd_lemma] += ele[1]
            expd_lemma2weight = dict(expd_lemma2weight)
            
            obj_head2embed_list[obj_head].append(obj_head_embed)
            obj_head2freq[obj_head] += 1
            if obj_head not in obj_head2expd_lemma_w_weight:
                obj_head2expd_lemma_w_weight[obj_head] = defaultdict(float)
            for expd_lemma, weight in expd_lemma2weight.items():
                obj_head2expd_lemma_w_weight[obj_head][expd_lemma] += weight

        processed_svo_cnt += 1  
    
    print(f'Processed {processed_svo_cnt} SVO triplets')
    sense2freq = dict(sense2freq)
    sense2embed = {k:np.average(v, axis=0) for k, v in sense2embed_list.items() if k in sense2embed_list}
    sense2expd_lemma_w_weight = {k: dict(v) for k, v in sense2expd_lemma_w_weight.items() if k in sense2expd_lemma_w_weight}
    obj_head2freq = dict(obj_head2freq)
    obj_head2embed = {k:np.average(v, axis=0) for k, v in obj_head2embed_list.items() if k in obj_head2freq}
    obj_head2expd_lemma_w_weight = {k: dict(v) for k, v in obj_head2expd_lemma_w_weight.items() if k in obj_head2expd_lemma_w_weight}

    print ('Getting seperate features for predicate senses and object heads...')
    selected_vs_list = list(sense2freq.keys())
    inv_vs_vocab = {vs:i for i, vs in enumerate(selected_vs_list)}
    selected_oh_list = list(obj_head2freq.keys())
    inv_oh_vocab = {oh:i for i, oh in enumerate(selected_oh_list)}

    selected_vs_expd_lemma_w_weight = {vs:sense2expd_lemma_w_weight[vs] for vs in selected_vs_list}
    print ('Get verb sense context expansion embedding...')
    vs_context_embed = tfidf_pca(selected_vs_list, selected_vs_expd_lemma_w_weight, sense2freq)
    print ('Get verb sense BERT embedding...')
    selected_vs_bert_embed = np.array([sense2embed[vs] for vs in selected_vs_list])
    vs_bert_embed = pca(selected_vs_bert_embed)
    print ('Merge and get verb sense final embedding...')  
    vs_final_embed = np.concatenate([vs_bert_embed, vs_context_embed], axis=1)

    selected_oh_expd_lemma_w_weight = {oh:obj_head2expd_lemma_w_weight[oh] for oh in selected_oh_list}
    print ('Get object head context expansion embedding...')
    oh_context_embed = tfidf_pca(selected_oh_list, selected_oh_expd_lemma_w_weight, obj_head2freq)
    print ('Get object head BERT embedding...')
    selected_oh_bert_embed = np.array([obj_head2embed[oh] for oh in selected_oh_list])
    oh_bert_embed = pca(selected_oh_bert_embed)
    print ('Merge and get verb sense final embedding...') 
    oh_final_embed = np.concatenate([oh_bert_embed, oh_context_embed], axis=1)

    # select the top-15 event types with the most matched results for both datasets to avoid types with too few mentions
    print ('Event Type Frequence...')
    etype_freq = {}
    for svo_id, svo_info in tqdm(po_mentions.items()):
        vs = f"{svo_info['verb']}_{sense_id}"
        oh = svo_info['obj_head']
        if vs not in selected_vs_list or oh not in selected_oh_list:
            continue
        if svo_info['event_type'] not in etype_freq:
            etype_freq[svo_info['event_type']] = 1
        else:
            etype_freq[svo_info['event_type']] += 1
    
    etype_freq = sorted(etype_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    etype_freq = [e[0] for e in etype_freq]
    print (etype_freq)

    print('Getting P-O Tuple Features...')
    vocab = {}
    inv_vocab = {}
    eventtype = {}
    vs_emb = []
    oh_emb = []
    vs_w_oh2freq = defaultdict(int)
    for svo_id, svo_info in tqdm(po_mentions.items()):
        sense_id = svo_id2sense_id[svo_id]
        vs = f"{svo_info['verb']}_{sense_id}"
        oh = svo_info['obj_head']
        if vs not in selected_vs_list or oh not in selected_oh_list or svo_info['event_type'] not in etype_freq:
            continue
        vs_oh_tuple = (vs, oh)
        if vs_oh_tuple not in vocab:
            index = len(vocab)
            vocab[vs_oh_tuple] = index
            inv_vocab[index] = vs_oh_tuple
            eventtype[index] = svo_info['event_type']
            vs_emb.append(vs_final_embed[inv_vs_vocab[vs],:])
            oh_emb.append(oh_final_embed[inv_oh_vocab[oh],:])
        else:
            if etype_freq.index(svo_info['event_type']) < etype_freq.index(eventtype[vocab[vs_oh_tuple]]):
               eventtype[vocab[vs_oh_tuple]] = svo_info['event_type']

        vs_w_oh2freq[vs_oh_tuple] += 1
        
    vs_w_oh2freq = dict(vs_w_oh2freq)
    vs_emb = np.array(vs_emb).astype(np.float32)
    oh_emb = np.array(oh_emb).astype(np.float32)
    tuple_freq = []
    for i in range(len(inv_vocab)):
        tuple_freq.append(vs_w_oh2freq[inv_vocab[i]])
    print (f'Total number of distinct (verb sense, object head) pairs: {len(vocab)}')

    with open(os.path.join(data_dir, 'corpus_parsed_svo_po_tuple_features_for_eval.pk'), 'wb') as f:
        pk.dump({
            'vocab': vocab,
            'inv_vocab': inv_vocab,
            'vs_emb': vs_emb,
            'oh_emb': oh_emb,
            'tuple_freq': tuple_freq,
            'event_type': eventtype,
        }, f)
    
    id_map = etype_freq
    golden = [id_map.index(v) for k, v in eventtype.items()]

    mention_clusters = [set() for _ in range(15)]
    for i, clus_num in enumerate(golden):
        tup = inv_vocab[i]
        if tup not in mention_clusters[clus_num]:
            mention_clusters[clus_num].add(tup)
    
    with open(os.path.join(data_dir, 'golden.json'), 'w') as f:
        for i, cluster in enumerate(mention_clusters):
            result_string = f"Topic {i} ({len(cluster)}) ({id_map[i]}): "
            for vo in cluster:
                result_string += ', ' + ' '.join([vo[0].split('_')[0], vo[1]])
            f.write(result_string + '\n')

def eval(data_dir, dict_file, gpu_id, num_sample):
    if not os.path.exists(os.path.join(data_dir, 'corpus_parsed_svo_po_tuple_features_for_eval.pk')):
        print('Loading corpus and salient term files...')
        with open(os.path.join(data_dir, 'corpus_parsed_svo.json')) as f:
            corpus = json.load(f)
        with open(os.path.join(data_dir, 'corpus_parsed_svo_salient_verbs.json')) as f:
            salient_verbs = json.load(f)
        with open(os.path.join(data_dir, 'corpus_parsed_svo_salient_obj_heads.json')) as f:
            salient_obj_heads = json.load(f)  
        with open(os.path.join(data_dir, 'corpus_parsed_svo_obj2obj_head_info.json')) as f:
            obj2obj_head_infos = json.load(f)
        with open(dict_file, 'r') as f:
            verb_sense_dict = json.load(f)
        with open(os.path.join(data_dir, 'train.oneie.json')) as f:
            original_corpus = f.readlines()

        print ('Prepare features for baselines... ')
        prepare_for_baselines(gpu_id, data_dir, corpus, salient_verbs, salient_obj_heads, obj2obj_head_infos, verb_sense_dict, original_corpus)

    data_dict = pk.load(open(os.path.join(data_dir, 'corpus_parsed_svo_po_tuple_features_for_eval.pk'), 'rb'))
    with open(os.path.join(data_dir, 'golden.json'), 'r') as f:
        id_map = [line.split()[3].strip('(').strip('):') for line in f.readlines()]
        golden = [id_map.index(v) for k, v in data_dict['event_type'].items()]
        print (len(golden))

    with open(os.path.join(data_dir, 'result.json'), 'w') as f:
        print ('Begin KMeans...')
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = kmeans(data_dict, id_map, 15, os.path.join(data_dir, 'kmeans_pred'))
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('kmeans\n')
        f.write(str(tb) + '\n')
    
        print ('Begin AggClus...')
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = agglo(data_dict, id_map, 15, os.path.join(data_dir, 'agglo_pred'))
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('AggClus\n')
        f.write(str(tb) + '\n')

        print ('Begin JCSC...')
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = jcsc(data_dict, id_map, 15, os.path.join(data_dir, 'jcsc_pred'))
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('JCSC\n')
        f.write(str(tb) + '\n')
        
        print ('Begin Triframes-ChineseWhispers...')
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = triframes(False, data_dict, id_map, os.path.join(data_dir, 'triframes_chinesewhispers_pred'))
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('Triframes-ChineseWhispers\n')
        f.write(str(tb) + '\n')

        print ('Begin Triframes-Watset...')
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = triframes(True, data_dict, id_map, os.path.join(data_dir, 'triframes_watset_pred'))
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('Triframes-Watset\n')
        f.write(str(tb) + '\n')

        print ('Begin ETypeClus ...')
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = etyeclus(data_dict, id_map, 15, os.path.join(data_dir, 'etyeclus_pred'))
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ETypeClus\n')
        f.write(str(tb) + '\n')

        print ('Begin ESHer...')
        with open(os.path.join('lm_output', 'ere')) as f_output:
            lm_outputs = ''.join(f_output.readlines())
            lm_outputs = lm_outputs.split('\n\n###\n\n')[:-1]
        with open('data/ace05/incontext-0.json') as f_incontext:
            incontexts = json.load(f_incontext)
            shot = len(incontexts)
        outputs = {}
        instance = []
        for output in lm_outputs:
            output = output.split('\n')[shot]
            vo = output.split('&')[0].strip()
            output = output.split('&')[1].strip().split()

            if len(output) == 0: 
                instance.append(['none', 'none'])
            elif len(output) == 1: 
                instance.append([output[0], 'none'])
            else: 
                instance.append([output[0]] + list(set(output[1:])))
            if len(instance) == num_sample:
                outputs[vo] = instance
                instance = []
        lm_outputs = []
        for k, v in data_dict['vocab'].items():
            k = ' '.join([k[0].split('_')[0], k[1]])
            lm_outputs.append((k, outputs[k])) 
        
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'verbmatch_pred'), intra=None, inter='VerbMatch', times=1)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('VerbMatch\n')
        f.write(str(tb) + '\n')

        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'esher_inter_pred'), intra=None, inter='TypeMatch', times=num_sample)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ESHer (inter w/o intra)\n')
        f.write(str(tb) + '\n')
        
        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'esher_inter_intra_pred'), intra=['TF-IDF', 'PageRank', 'WordSense'], inter='TypeMatch', times=num_sample)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ESHer (inter w intra)\n')
        f.write(str(tb) + '\n')

        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'esher_inter_intra_-WS_pred'), intra=['TF-IDF', 'PageRank'], inter='TypeMatch', times=num_sample)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ESHer (inter w intra) - WordSense\n')
        f.write(str(tb) + '\n')

        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'esher_inter_intra_-TI_pred'), intra=['PageRank', 'WordSense'], inter='TypeMatch', times=num_sample)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ESHer (inter w intra) - TF-IDF\n')
        f.write(str(tb) + '\n')

        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'esher_inter_intra_-PR_pred'), intra=['TF-IDF', 'WordSense'], inter='TypeMatch', times=num_sample)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ESHer (inter w intra) - PageRank\n')
        f.write(str(tb) + '\n')

        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'esher_inter_intra_-TIWS_pred'), intra=['PageRank'], inter='TypeMatch', times=num_sample)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ESHer (inter w intra) - TF-IDF - WordSense\n')
        f.write(str(tb) + '\n')

        ARI, NMI, ACC, Bcubed_F1, Num_Cluster, ARI_std, NMI_std, ACC_std, Bcubed_F1_std, Num_Cluster_std = esher(lm_outputs, data_dict, id_map, os.path.join(data_dir, 'esher_inter_intra_-PRWS_pred'), intra=['TF-IDF'], inter='TypeMatch', times=num_sample)
        tb = pt.PrettyTable()
        tb.field_names = ['ARI (std)', 'NMI (std)', 'ACC (std)', 'BCubed-F1 (std)', 'Num of Clusters']
        tb.add_row(['%2.2f (%1.2f)'%(ARI*100, ARI_std*100), '%2.2f (%1.2f)'%(NMI*100, NMI_std*100), '%2.2f (%1.2f)'%(ACC*100, ACC_std*100), '%2.2f (%1.2f)'%(Bcubed_F1*100, Bcubed_F1_std*100), '%2.2f (%1.2f)'%(Num_Cluster, Num_Cluster_std)])
        f.write('ESHer (inter w intra) - PageRank - WordSense\n')
        f.write(str(tb) + '\n')

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ere_dir', type=str, help='input corpus dir')
    parser.add_argument('--dict_file', type=str, help='dict file')
    parser.add_argument('--gpu_id', type=str, help='gpu id')
    parser.add_argument('--num_sample', type=int, help='sample of decode')
    args = parser.parse_args()
    print(vars(args))

    eval(args.ere_dir, args.dict_file, args.gpu_id, args.num_sample)
    print('Done!')