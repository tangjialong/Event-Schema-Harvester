import os
import json
import random
from tqdm import tqdm

import openai
openai.api_key = "XXXXX"

def get_in_context(incontext_corpus):
    exists_schema = {}
    incontext_instances = []

    if 'ace' in incontext_corpus:
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
        with open(os.path.join(incontext_corpus, 'schema.json'), 'w') as f:
            json.dump(exists_schema, f, ensure_ascii=False, sort_keys=True, indent=True)

        incontexts = {k.split(':')[0]: list() for k, v in exists_schema.items()}
        # sentence & eventtype role1 role2 role3 ...
        for instance in tqdm(incontext_instances):
            for event_mention in instance['event_mentions']:
                event_type = event_mention['event_type']
                incontext = [' '.join(instance['tokens']), '&'] + [event_type.split(':')[1]] + exists_schema[event_type]
                incontext = ' '.join(incontext).lower()
                incontexts[event_type.split(':')[0]].append(incontext)
    
    elif 'duee' in incontext_corpus:
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
        with open(os.path.join(incontext_corpus, 'schema.json'), 'w') as f:
            json.dump(exists_schema, f, ensure_ascii=False, sort_keys=True, indent=True)

        incontexts = {k.split('-')[0]: list() for k, v in exists_schema.items()}
        # sentence & eventtype role1 role2 role3 ...
        for instance in tqdm(incontext_instances):
            for event_mention in instance['event_list']:
                event_type = event_mention['event_type']
                incontext = [instance['text'], '&'] + [event_type.split('-')[1]] + exists_schema[event_type]
                incontext = ' '.join(incontext).lower()
                incontexts[event_type.split('-')[0]].append(incontext)

    with open(os.path.join(incontext_corpus, 'incontext.json'), 'w') as f:
        json.dump(incontexts, f, ensure_ascii=False, sort_keys=True, indent=True)
    return incontexts

def prepare_in_context(incontext_corpus, data=None):
    if not os.path.exists(os.path.join(incontext_corpus, 'incontext.json')):
        incontexts = get_in_context(incontext_corpus)
    else:
        with open(os.path.join(incontext_corpus, 'incontext.json')) as f:
            incontexts = json.load(f)

    shot = len(incontexts)

    selected_event_types = random.sample(incontexts.keys(), shot)
    output = ''
    for selected_event_type in selected_event_types:
        selected_incontext = '\n'.join(random.sample(incontexts[selected_event_type], 1))
        output = output + selected_incontext + '\n'
    return output

def generate(input, incontext_corpus, length=64):
    incontext = prepare_in_context(incontext_corpus)
    prompt = incontext + f'{input} &'
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=length,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    result = response["choices"][0]["text"].split('\n')[0].strip().split()
    event_type = result[0]
    event_slot = result[1:]
    # print(prompt)
    # print('\n')
    # print(event_type)
    # print('\n')
    # print(event_slot)
    # print('\n')
    return prompt, event_type, event_slot

if __name__ == "__main__":
    random.seed(1234)
