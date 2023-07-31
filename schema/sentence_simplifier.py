# coding=utf-8

import os
import json
import string
import argparse
from tqdm import tqdm

from collections import defaultdict
from collections.abc import Iterable

from nltk.corpus import stopwords

import spacy
from spacy.tokens import Doc

# words that are negations
NEGATIONS = {'no', 'not', 'n\'t', 'never', 'none'}
# dependency markers for subjects
SUBJECTS = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl'}
# dependency markers for objects
OBJECTS = {'dobj', 'dative', 'attr', 'oprd'}
# POS tags that will break adjoining items
BREAKER_POS = {'CCONJ', 'VERB'}

# is the token a verb?  (including auxiliary verbs)
def _is_verb(tok):
    return tok.pos_ == 'VERB' or tok.pos_ == 'AUX'
    
# is the token an auxiliary verb?
def _is_non_aux_verb(tok):
    return tok.pos_ == 'VERB' and (tok.dep_ != 'aux' and tok.dep_ != 'auxpass')

# find the main verb - or any aux verb if we can't find it
def _find_verbs(tokens):
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    if len(verbs) == 0:
        verbs = [tok for tok in tokens if _is_verb(tok)]
    return verbs

# return true if the current verb is passive
def _is_cur_v_passive(v):
    for tok in v.children:
        if tok.dep_ == 'auxpass':
            return True
    return False

# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False

# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet, depTagSet):
    lexical_match = ('and' in depSet or 'or' in depSet or 'nor' in depSet or \
           'but' in depSet or 'yet' in depSet or 'so' in depSet or 'for' in depSet)
    dependency_match = ('conj' in depTagSet)
    return lexical_match or dependency_match
    
# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        rightDepTags = {tok.dep_ for tok in rights}
        if contains_conj(rightDeps, rightDepTags):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == 'NOUN' or tok.dep_ == 'conj'])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs

# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != 'VERB' and head.pos_ != 'NOUN' and head.head != head:
        head = head.head
    if head.pos_ == 'VERB':
        subs = [tok for tok in head.lefts if tok.dep_ == 'SUB']
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == 'NOUN':
        return [head], _is_negated(tok)
    return [], False

# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != 'DET']
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated

# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v

# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == 'ADP' and (dep.dep_ == 'prep' or (is_pas and dep.dep_ == 'agent')):
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == 'PRON' and tok.lower_ == 'me') or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs

# xcomp; open complement - verb has no suject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == 'VERB' and dep.dep_ == 'xcomp':
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None

def contains_conj_strict(depSet, depTagSet):
    lexical_match = ('and' in depSet or 'or' in depSet or 'nor' in depSet or \
           'but' in depSet or 'yet' in depSet or 'so' in depSet or 'for' in depSet)
    return lexical_match

# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        rightDepTags = {tok.dep_ for tok in rights}
        if contains_conj_strict(rightDeps, rightDepTags):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == 'NOUN'])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs

# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs

# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return None

# expand an obj / subj np using its chunk
def expand(item, tokens, visited):
    if item.lower_ == 'that':
        temp_item = _get_that_resolution(tokens)
        if temp_item is not None:
            item = temp_item

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == 'DET' or item2.pos_ == 'NOUN':
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts

# convert a list of tokens to a string
def to_str(tokens):
    if isinstance(tokens, Iterable):
        return [' '.join([item.text for item in tokens]), [item.i for item in tokens]]
    else:
        return ['', [-1]]

# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    svos = []
    verbs = _find_verbs(tokens)
    visited = set()  # recursion detection
    for v in verbs:
        is_pas = _is_cur_v_passive(v)
        subs, verbNegated = _get_all_subs(v)
        # hopefully there are subs
        if len(subs) > 0:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        expanded_sub = expand(sub, tokens, visited)
                        expanded_obj = expand(obj, tokens, visited)
                        normalized_verb1 = '!' + v.lemma_ if verbNegated or objNegated else v.lemma_
                        normalized_verb2 = '!' + v2.lemma_ if verbNegated or objNegated else v2.lemma_
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expanded_obj), [normalized_verb1, v.i], to_str(expanded_sub)))
                            svos.append((to_str(expanded_obj), [normalized_verb2, v2.i], to_str(expanded_sub)))
                        else:
                            svos.append((to_str(expanded_sub), [normalized_verb1, v.i],  to_str(expanded_obj)))
                            svos.append((to_str(expanded_sub), [normalized_verb2, v2.i],  to_str(expanded_obj)))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    if len(objs) > 0:
                        for obj in objs:
                            objNegated = _is_negated(obj)
                            expanded_sub = expand(sub, tokens, visited)
                            expanded_obj = expand(obj, tokens, visited)
                            normalized_verb = '!' + v.lemma_ if verbNegated or objNegated else v.lemma_
                            if is_pas:  # reverse object / subject for passive
                                svos.append((to_str(expanded_obj), [normalized_verb, v.i], to_str(expanded_sub)))
                            else:
                                svos.append((to_str(expanded_sub), [normalized_verb, v.i], to_str(expanded_obj)))
                    else:
                        # no obj - just return the SV parts
                        expanded_sub = expand(sub, tokens, visited)
                        normalized_verb = '!' + v.lemma_ if verbNegated else v.lemma_
                        if is_pas:
                            svos.append((None, [normalized_verb, v.i], to_str(expanded_sub)))
                        else:
                            svos.append((to_str(expanded_sub), [normalized_verb, v.i], None))
        else:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for obj in objs:
                    objNegated = _is_negated(obj)
                    expanded_obj = expand(obj, tokens, visited)
                    normalized_verb1 = '!' + v.lemma_ if verbNegated or objNegated else v.lemma_
                    normalized_verb2 = '!' + v2.lemma_ if verbNegated or objNegated else v2.lemma_
                    
                    if is_pas:  # reverse object / subject for passive
                        svos.append((to_str(expanded_obj), [normalized_verb1, v.i], None))
                        svos.append((to_str(expanded_obj), [normalized_verb2, v2.i], None))
                    else:
                        svos.append((None, [normalized_verb1, v.i], to_str(expanded_obj)))
                        svos.append((None, [normalized_verb2, v2.i], to_str(expanded_obj)))
            else:
                v, objs = _get_all_objs(v, is_pas)
                if len(objs) > 0:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        expanded_obj = expand(obj, tokens, visited)
                        normalized_verb = '!' + v.lemma_ if verbNegated or objNegated else v.lemma_
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expanded_obj), [normalized_verb, v.i], None))
                        else:
                            svos.append((None, [normalized_verb, v.i], to_str(expanded_obj)))

    return svos

def parse_corpus_and_extract_svo(input_file,
                                 spacy_model):
    print('Loading Spacy model...')
    nlp = spacy.load(spacy_model)

    print('Loading Corpus...')
    fin = open(input_file, 'r')
    lines = fin.readlines()
    
    save_dict_data = {}
    for id, line in enumerate(tqdm(lines)):
        line = json.loads(line)
        sent = line['text'].lower()
        doc = nlp(sent)
        for token in doc: # for chinese
            if len(token.lemma_) == 0: token.lemma_ = token.text
        token_list = [token.text for token in doc]
        raw_sent = ' '.join(token_list)
        svos = findSVOs(doc)
        save_dict_data[id] = {
            'sent_id': id,
            'raw_sentence': raw_sent,
            'token_list': token_list,
            'svos': svos,
        }

    save_path = os.path.join(os.path.dirname(input_file), 'corpus_parsed_svo.json')
    with open(save_path, 'w') as f:
        json.dump(save_dict_data, f, ensure_ascii=False, sort_keys=True, indent=True)

    return save_path

####################################################################################################

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def get_salience(item2local_freq, item2global_freq):
    import math
    N_sent_bkg = 100000000 #TODO
    item2salience = {}
    for item, local_freq in item2local_freq.items():
        if item not in item2global_freq:
            item2salience[item] = -1
        else:
            salience = (1+math.log(local_freq, 10)**2)*math.log(N_sent_bkg/item2global_freq[item] , 10)
            item2salience[item] = salience
    return item2salience

stop_words = set(stopwords.words('english'))
for c in string.ascii_letters:
    stop_words.add(c)
for c in string.digits:
    stop_words.add(c)
for c in string.punctuation:
    stop_words.add(c)

def get_salient_frequent_verb_lemmas(verb2local_freq, verb2global_freq, top_ratio=0.8, min_freq=5):
    verb2salience = get_salience(verb2local_freq, verb2global_freq)    
    stopword_verbs = stop_words | {
        'could', 'can', 'may', 'might', 'will', 'would', 'should', 'shall', 'be',
        '\'d\'', ',', '’', 'take', 'use', 'make', 'have', 'go', 'come', 'get', 'do',
        'give', 'put', 'set', 'argue', 'say', 'claim', 'suggest', 'tell', 
    } 

    V = int(len(verb2salience) * top_ratio)
    salient_verbs = {}
    for ele in sorted(verb2salience.items(), key=lambda x:-x[1]):
        if ele[0] not in stopword_verbs:
            salient_verbs[ele[0]] = ele[1]
        if len(salient_verbs) == V:
            break
            
    print(f'Select {len(salient_verbs)} salient verbs')

    frequent_salient_verbs = {}
    for verb, saliency in salient_verbs.items():
        if verb2local_freq[verb] >= min_freq:
            frequent_salient_verbs[verb] = saliency
        
    print(f'Select {len(frequent_salient_verbs)} frequent and salient verbs')
    return frequent_salient_verbs

def get_salient_frequent_object_heads(oh2local_freq, oh2global_freq, top_ratio=0.8, min_freq=3):
    oh2salience = get_salience(oh2local_freq, oh2global_freq)    

    stopword_nouns = stop_words | {''}
    V = int(len(oh2salience) * top_ratio)
    salient_oh = {}
    for ele in sorted(oh2salience.items(), key=lambda x:-x[1]):
        if ele[0] not in stopword_nouns:
            salient_oh[ele[0]] = ele[1]
        if len(salient_oh) == V:
            break
            
    print(f'Select {len(salient_oh)} salient object heads')

    frequent_salient_ohs = {}
    for verb, saliency in salient_oh.items():
        if oh2local_freq[verb] >= min_freq:
            frequent_salient_ohs[verb] = saliency
        
    print(f'Select {len(frequent_salient_ohs)} frequent and salient object heads')
    return frequent_salient_ohs

def select_salient_terms(corpus_w_svo_pickle,
                         verb_freq_file,
                         all_lemma_freq_file,
                         spacy_model,
                         min_verb_freq,
                         top_verb_ratio,
                         min_obj_freq,
                         top_obj_ratio):
    print('Loading Spacy model...')
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    print('Loading Corpus...')
    with open(corpus_w_svo_pickle, 'rb') as f:
        corpus = json.load(f)

    print('Loading term frequency files...')
    with open(verb_freq_file, 'r') as f:
        verb2global_freq = json.load(f)
    with open(all_lemma_freq_file, 'rb') as f:
        lemma2global_freq = json.load(f)

    verb2local_freq = defaultdict(int)
    obj2local_freq = defaultdict(int)
    for doc in corpus.values():
        for em in doc['svos']:
            verb = em[1][0]
            if verb.startswith('!'):
                verb = verb[1:]
            verb2local_freq[verb] += 1
            if em[2] is not None:
                obj = em[2][0]
                obj2local_freq[obj] += 1
    
    verb2local_freq = dict(verb2local_freq)
    obj2local_freq = dict(obj2local_freq)

    print('Obtain object heads...')
    obj_head2local_freq = defaultdict(int)
    obj2obj_head_info = {}
    for obj, local_freq in tqdm(obj2local_freq.items()):
        if ' '.join(obj.split()) != obj:
            continue
        parsed_obj = nlp(obj)
        for token in parsed_obj: # for chinese
            if len(token.lemma_) == 0: token.lemma_ = token.text
        for i, tok in enumerate(parsed_obj):
            if tok.dep_ == 'ROOT':
                obj_head_lemma = tok.lemma_
                obj_head_relative_index = i
                break
        obj2obj_head_info[obj] = {
            'obj_head_lemma': obj_head_lemma,
            'obj_head_relative_index': obj_head_relative_index,
        }
        obj_head2local_freq[obj_head_lemma] += local_freq
    obj_head2local_freq = dict(obj_head2local_freq)

    frequent_salient_verbs = get_salient_frequent_verb_lemmas(
        verb2local_freq, verb2global_freq, top_ratio=top_verb_ratio, min_freq=min_verb_freq)
    
    frequent_salient_object_heads = get_salient_frequent_object_heads(
        obj_head2local_freq, lemma2global_freq, top_ratio=top_obj_ratio, min_freq=min_obj_freq)

    print('Saving selected salient terms...')
    with open(f'{corpus_w_svo_pickle[:-5]}_salient_verbs.json', 'w') as f:
        json.dump(frequent_salient_verbs, f, ensure_ascii=False, sort_keys=True, indent=True)
    with open(f'{corpus_w_svo_pickle[:-5]}_salient_obj_heads.json', 'w') as f:
        json.dump(frequent_salient_object_heads, f, ensure_ascii=False, sort_keys=True, indent=True)
    with open(f'{corpus_w_svo_pickle[:-5]}_obj2obj_head_info.json', 'w') as f:
        json.dump(obj2obj_head_info, f, ensure_ascii=False, sort_keys=True, indent=True)

####################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy_model', default='en_core_web_lg', help='Spacy model for POS tagging')

    # parse_corpus_and_extract_svo
    parser.add_argument('--input_file', help='input corpus file')

    # select_salient_terms
    parser.add_argument('--verb_freq_file', default='./resources/en-etypeclus/verb_freq.json')
    parser.add_argument('--all_lemma_freq_file', default='./resources/en-etypeclus/all_lemma_freq.json')
    parser.add_argument('--min_verb_freq', default=3, type=int, help='minimum verb lemma frequency')
    parser.add_argument('--top_verb_ratio', default=0.25, type=float, help='top percentage of selected salient verb lemma')
    parser.add_argument('--min_obj_freq', default=3, type=int, help='minimum object head frequency')
    parser.add_argument('--top_obj_ratio', default=0.25, type=float, help='top percentage of selected salient object head')

    args = parser.parse_args()
    print(vars(args))

    save_path = parse_corpus_and_extract_svo(args.input_file, args.spacy_model)
    select_salient_terms(save_path, args.verb_freq_file, args.all_lemma_freq_file, args.spacy_model, args.min_verb_freq, args.top_verb_ratio, args.min_obj_freq, args.top_obj_ratio)
    print('Done!')