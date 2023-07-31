# coding=utf-8

import os
import json
import argparse
from tqdm import tqdm

import spacy

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

def create_resources(background_corpus):
    print('Loading Spacy model...')
    if 'en' in background_corpus:
        spacy_model = 'en_core_web_lg'
        all_lemma_freq_file = os.path.join(os.path.dirname(background_corpus), 'en_all_lemma_freq.json')
        verb_lemma_freq_file = os.path.join(os.path.dirname(background_corpus), 'en_verb_lemma_freq.json')
    else:
        spacy_model = 'zh_core_web_lg'
        all_lemma_freq_file = os.path.join(os.path.dirname(background_corpus), 'zh_all_lemma_freq.json')
        verb_lemma_freq_file = os.path.join(os.path.dirname(background_corpus), 'zh_verb_lemma_freq.json')
    nlp = spacy.load(spacy_model)

    all_lemma_freq = {}
    verb_lemma_freq = {}

    print('Loading Background Corpus...')
    fin = open(background_corpus, 'r')
    lines = fin.readlines()
    for i, line in enumerate(tqdm(lines)):
        line = json.loads(line)
        if 'en' in background_corpus:
            sent = line['text'].lower()
        else:
            sent = line['text']
        tokens = nlp(sent)
        verbs = _find_verbs(tokens)
        
        for token in tokens:
            token = token.lemma_ if 'en' in background_corpus else token.text
            if token not in all_lemma_freq:
                all_lemma_freq[token] = 1
            else:
                all_lemma_freq[token] += 1
        for verb in verbs:
            verb = verb.lemma_ if 'en' in background_corpus else verb.text
            if verb not in verb_lemma_freq:
                verb_lemma_freq[verb] = 1
            else:
                verb_lemma_freq[verb] += 1

    with open(all_lemma_freq_file, 'w') as f:
        json.dump(all_lemma_freq, f, ensure_ascii=False, sort_keys=True, indent=True)
    with open(verb_lemma_freq_file, 'w') as f:
        json.dump(verb_lemma_freq, f, ensure_ascii=False, sort_keys=True, indent=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enwiki', default='resources/enwiki.json')
    parser.add_argument('--zhwiki', default='resources/zhwiki.json')
    args = parser.parse_args()
    print(vars(args))
    
    create_resources(args.enwiki)
    create_resources(args.zhwiki)

    print('Done')