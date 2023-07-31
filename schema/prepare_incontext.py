# coding=utf-8

import os
import json
import random
import argparse
from tqdm import tqdm

import spacy

def get_in_context(incontext_corpus, mode):
    exists_schema = {}
    incontext_instances = []

    if 'ace' in incontext_corpus:
        if mode == '0':
            print('Loading Spacy model...')
            nlp = spacy.load('en_core_web_lg')
            with open(os.path.join(incontext_corpus, 'train.oneie.json')) as f:
                for line in f:
                    line = json.loads(line)
                    incontext_instances.append(line)
                    
                    for event_mention in line['event_mentions']:
                        event_type = event_mention['event_type']
                        if event_type not in exists_schema:
                            exists_schema[event_type] = set()
                        for argument in event_mention['arguments']:
                            exists_schema[event_type].add(argument['role'])
            
            exists_schema = {k: list(v) for k, v in exists_schema.items()}

            incontexts = {k.split(':')[0]: list() for k, v in exists_schema.items()}
            # verb object & eventtype role1 role2 role3 ...
            for instance in tqdm(incontext_instances):
                for event_mention in instance['event_mentions']:
                    event_type = event_mention['event_type']
                    verb_lemma = nlp(event_mention['trigger']['text'])[0].lemma_
                    for argument in event_mention['arguments']:
                        for entity in instance['entity_mentions']:
                            if entity['id'] == argument['entity_id'] and entity['mention_type'] != 'PRO':
                                parsed_obj = nlp(argument['text'])
                                for i, tok in enumerate(parsed_obj): # find head
                                    if tok.dep_ == 'ROOT':
                                        obj_head_lemma = tok.lemma_
                                        break
                                incontext = [verb_lemma, obj_head_lemma, '&'] + [event_type.split(':')[1]] + exists_schema[event_type]
                                incontext = ' '.join(incontext).lower()
                                incontexts[event_type.split(':')[0]].append(incontext)
        elif mode == '1':
            with open(os.path.join(incontext_corpus, 'train.oneie.json')) as f:
                for line in f:
                    line = json.loads(line)
                    incontext_instances.append(line)
                    
                    for event_mention in line['event_mentions']:
                        event_type = event_mention['event_type']
                        if event_type not in exists_schema:
                            exists_schema[event_type] = set()
                        for argument in event_mention['arguments']:
                            exists_schema[event_type].add(argument['role'])
            
            exists_schema = {k: list(v) for k, v in exists_schema.items()}

            incontexts = {k.split(':')[0]: list() for k, v in exists_schema.items()}
            # sentence & eventtype role1 role2 role3 ...
            for instance in tqdm(incontext_instances):
                for event_mention in instance['event_mentions']:
                    event_type = event_mention['event_type']
                    incontext = [' '.join(instance['tokens']), '&'] + [event_type.split(':')[1]] + exists_schema[event_type]
                    incontext = ' '.join(incontext).lower()
                    incontexts[event_type.split(':')[0]].append(incontext)

    elif 'duee' in incontext_corpus:
        if mode == '0':
            print('Loading Spacy model...')
            nlp = spacy.load('zh_core_web_lg')
            with open(os.path.join(incontext_corpus, 'train.json')) as f:
                for line in f:
                    line = json.loads(line)
                    incontext_instances.append(line)
        
                    for event_mention in line['event_list']:
                        event_type = event_mention['event_type']
                        if event_type not in exists_schema:
                            exists_schema[event_type] = set()
                        for argument in event_mention['arguments']:
                            exists_schema[event_type].add(argument['role'])

            exists_schema = {k: list(v) for k, v in exists_schema.items()}

            incontexts = {k.split('-')[0]: list() for k, v in exists_schema.items()}
            # verb object & eventtype role1 role2 role3 ...
            for instance in tqdm(incontext_instances):
                for event_mention in instance['event_list']:
                    event_type = event_mention['event_type']
                    verb_lemma = event_mention['trigger']
                    for argument in event_mention['arguments']:
                        if len(argument['argument']) > 32: continue # too long argument
                        parsed_obj = nlp(argument['argument'])
                        for i, tok in enumerate(parsed_obj): # find head
                            if tok.dep_ == 'ROOT':
                                obj_head_lemma = tok.text
                                break
                        incontext = [verb_lemma, obj_head_lemma, '&'] + [event_type.split('-')[1]] + exists_schema[event_type]
                        incontext = ' '.join(incontext).lower()
                        incontexts[event_type.split('-')[0]].append(incontext)
        elif mode == '1':
            with open(os.path.join(incontext_corpus, 'train.json')) as f:
                for line in f:
                    line = json.loads(line)
                    incontext_instances.append(line)
        
                    for event_mention in line['event_list']:
                        event_type = event_mention['event_type']
                        if event_type not in exists_schema:
                            exists_schema[event_type] = set()
                        for argument in event_mention['arguments']:
                            exists_schema[event_type].add(argument['role'])

            exists_schema = {k: list(v) for k, v in exists_schema.items()}

            incontexts = {k.split('-')[0]: list() for k, v in exists_schema.items()}
            # sentence & eventtype role1 role2 role3 ...
            for instance in tqdm(incontext_instances):
                for event_mention in instance['event_list']:
                    event_type = event_mention['event_type']
                    incontext = [instance['text'], '&'] + [event_type.split('-')[1]] + exists_schema[event_type]
                    incontext = ' '.join(incontext).lower()
                    incontexts[event_type.split('-')[0]].append(incontext)

    with open(os.path.join(incontext_corpus, f'incontext-{mode}.json'), 'w') as f:
        json.dump(incontexts, f, ensure_ascii=False, sort_keys=True, indent=True)
    return incontexts

def prepare_in_context(incontext_corpus, data, mode, top_num=500):
    if not os.path.exists(os.path.join(incontext_corpus, f'incontext-{mode}.json')):
        incontexts = get_in_context(incontext_corpus, mode)
    else:
        with open(os.path.join(incontext_corpus, f'incontext-{mode}.json')) as f:
            incontexts = json.load(f)
    
    shot = len(incontexts)

    with open(os.path.join(data, os.path.basename(data)+'-incontext.json'), 'w') as fout:
        outputs = []
        with open(os.path.join(data, 'corpus_parsed_svo.json')) as f:
            corpus_parsed_svo = json.load(f)

        if mode == '0':
            with open(os.path.join(data, 'corpus_parsed_svo_salient_verbs.json')) as f:
                salient_verbs = json.load(f)
            with open(os.path.join(data, 'corpus_parsed_svo_salient_obj_heads.json')) as f:
                salient_obj_heads = json.load(f)  
            with open(os.path.join(data, 'corpus_parsed_svo_obj2obj_head_info.json')) as f:
                obj2obj_head_infos = json.load(f)     
            
            vos = []
            for k, v in corpus_parsed_svo.items():
                for svo in v['svos']:
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
                                    vos.append([verb_lemma, obj_head_lemma, salient_verbs[verb_lemma]*salient_obj_heads[obj_head_lemma]])
            if top_num == 0:
                vos = sorted(list(set(tuple(sub) for sub in vos)), key=lambda x: x[2])
            else:
                vos = sorted(list(set(tuple(sub) for sub in vos)), key=lambda x: x[2], reverse=True)[:top_num]
            for vo in vos:
                selected_event_types = random.sample(incontexts.keys(), shot)
                output = ''
                for selected_event_type in selected_event_types:
                    selected_incontext = '\n'.join(random.sample(incontexts[selected_event_type], 1))
                    output = output + selected_incontext + '\n'
                output = output + f'{vo[0]} {vo[1]} &\n'
                outputs.append(output)
            fout.write('\n'.join(outputs))
            print(len(vos))

        elif mode == '1':
            count = 0
            for k, v in corpus_parsed_svo.items():
                selected_event_types = random.sample(incontexts.keys(), shot)
                output = ''
                for selected_event_type in selected_event_types:
                    selected_incontext = '\n'.join(random.sample(incontexts[selected_event_type], 1))
                    output = output + selected_incontext + '\n'
                raw_sentence = v['raw_sentence'].replace(' ', '')
                output = output + f'{raw_sentence} &\n'
                outputs.append(output)
                count += 1    
            fout.write('\n'.join(outputs))

            print(count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--incontext_corpus', type=str, default='data/duee')
    parser.add_argument('--data', type=str, default='data/covid19')
    parser.add_argument('--mode', type=str, default='0')
    parser.add_argument('--top_num', type=int, default=500)
    args = parser.parse_args()
    print(vars(args))

    random.seed(1234)
    prepare_in_context(args.incontext_corpus, args.data, args.mode, args.top_num)

    print('Done!')