# coding=utf-8

import re
import os
import json
import nltk
import opencc
import random
import argparse
from tqdm import tqdm
from collections import Counter

def clean_sen(sen):
    return sen.replace('�', '').replace('', '').replace('', '').replace('', '')

def zh_sen_tokenizer(text):
    ret = [(m.start(), m.end()) for m in re.finditer(r'[。]|[!?！？]+', text)]
    if len(ret) == 0:
        return [(0, len(text))]
    sentence_offset_list = list()
    pre_end = 0
    for start, end in ret:
        sentence_offset_list += [(pre_end, end)]
        pre_end = end

    sentence_list = []
    for (b,e) in sentence_offset_list:
        sentence_list.append(text[b:e])
    return sentence_list

def split_text_sen(input_dir, output_dir, jsonline_dir, data):
    converter = opencc.OpenCC('t2s.json')

    error_begin = 0
    error_string = []
    parag_name = []
    sentence_len = []
    stange_bigger = []
    ann = []

    all_file_path = []
    ann_f = open(os.path.join(jsonline_dir, data+'-5Ksample.json'), 'w')
    jsonline_f = open(os.path.join(jsonline_dir, data+'.json'), 'w')

    for f_dir in os.listdir(input_dir):
        for file_name in os.listdir(os.path.join(input_dir, f_dir)):
            all_file_path.append((os.path.join(input_dir, f_dir, file_name), os.path.join(output_dir, f_dir, file_name)))

    for (input_file_path, output_file_path) in tqdm(all_file_path):
        save_dir, _ = os.path.split(output_file_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        in_f = open(input_file_path)
        out_f = open(output_file_path, 'w')
        text_data = in_f.readlines()

        if 'en' in data:
            sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            for line in text_data:
                line = json.loads(line)
                if line['text'].strip().startswith('is ') or line['text'].strip().startswith('was ') or \
                    line['text'].strip().startswith('were ') or line['text'].strip().startswith('are '):
                    error_begin += 1
                    error_string.append(' '.join(line['text'].strip().split()[:10]))
                    line['text'] = line['title'].strip() + ' ' + line['text'].strip()
                text = line['text'].strip().split('\n')
                paragraphs = []
                for parag in text:
                    if len(parag.strip().split()) <= 5:
                        parag_name.append(parag.strip())
                    else:
                        paragraphs.append(parag.strip())    
                sentences = sentence_tokenizer.tokenize(' '.join(paragraphs))
                sentence_len += [len(s.strip().split()) for s in sentences]
                line['text'] = sentences
                ann += sentences
                out_f.write(json.dumps(line, ensure_ascii=False) + '\n')

        if 'zh' in data:
            sentence_tokenizer = zh_sen_tokenizer
            for line in text_data:
                line = json.loads(line)
                line['text'] = converter.convert(line['text']) 
                if line['text'].strip().startswith('是') and line['text'][:2] != line['title'][:2]:
                    line['text'] = line['title'] + line['text']
                text = line['text'].strip().split('\n')
                paragraphs = []
                for parag in text:
                    if len(parag) <= 5:
                        parag_name.append(parag.strip())
                    else:
                        paragraphs.append(parag.strip())
                sentences = sentence_tokenizer(''.join(paragraphs))
                stange_bigger += [len(s) for s in sentences if len(s) > 250]
                sentence_len += [len(s) for s in sentences]
                if type(sentences[0]) is str:
                    line['text'] = sentences
                    ann += sentences
                out_f.write(json.dumps(line, ensure_ascii=False) + '\n')

    random.shuffle(ann)
    for sentence in ann[:5000]:
        ann_f.write(json.dumps({'text': clean_sen(sentence)}, ensure_ascii=False) + '\n')
    for sentence in ann:
        jsonline_f.write(json.dumps({'text': clean_sen(sentence)}, ensure_ascii=False) + '\n')

    print('sentence_len (min/max/ave) {}/{}/{}'.format(min(sentence_len),
                                                       max(sentence_len),
                                                       sum(sentence_len) / len(sentence_len)))
    print(Counter(sentence_len).most_common(10))
    print('stange_bigger: ', len(stange_bigger), Counter(stange_bigger).most_common(10))
    print('parag_name {}/{}'.format(len(parag_name), len(set(parag_name))))
    print(Counter(parag_name).most_common(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', type=str, default=None, required=True, help='Json dir')
    parser.add_argument('-o', dest='output_dir', type=str, default=None, help='Json output dir')
    parser.add_argument('-j', dest='jsonline_dir', type=str, default=None, required=True, help='Json line output file')
    parser.add_argument('-d', dest='data', type=str, default='enwiki', choices=['enwiki', 'zhwiki'], help='Raw data')
    args = parser.parse_args()

    split_text_sen(args.input_dir, args.output_dir, args.jsonline_dir, args.data)

    print ('Done!')

if __name__ == '__main__':
    random.seed(1234)
    main()