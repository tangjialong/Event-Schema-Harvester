# coding=utf-8

import os
import re
import json
import nltk
import opencc
import argparse
from tqdm import tqdm
from xml.etree.ElementTree import parse

def text_clean(text):
	text = str(text)
	bad_str = ['\u2022', '\xf1', '\u2002', '\xae', '\u200b', '\ue6ce', '\ue63c',
	'\u200d', '\xdc', '\ufffc', '\u25b6', '\u25b7', '\u261f', '\uff64',
	'\u25d0', '\xb3', '\u27a4', '\u2776', '\u4a3b', '\u2665', '\xb2', '\ub9ac',
	'\u10e6', '\xb4', '\uff65', '\u1d17', '\u2ed3', '\u2590', '\u2003', '\ud734',
	'\u2f94', '\u22ef', '\u30fb', '\xd6', '\u2f6f', '\u2f08', '\u2f2f', '\u20e3',
	'\u2f5a', '\u2f00', '\ufffd', '\uf0fc', '\u207a', '\u25fe', '\u25ba', '\ucd1d',
	'\u301c', '\uec69', '\uede0', '\uedc6', '\u2fa6', '\u2fb0', '\u2f64', '\U0001f920',
	'\u2f8f', '\u2f47', '\u2edb', '\u2f79', '\uf06c', '\ufe0f', '\ue60a', '\ue404',
	'\u2122', '\ud55c', '\uac15', '\uae40', '\ud604', '\uc6b1', '\u2113', '\U0001f929',
	'\uec49', '\uee07', '\uec65', '\ued8c', '\uecfd', '\ued70', '\ued33', '\u2084',
	'\ued94', '\xf6', '\u25aa', '\u2f29', '\ufeff', '\uf0d6', '\uf0f0', '\xa0', '\u31cf',
	'\u25e6', '\u2ec4', '\uff61', '\u2b06', '\xa3', '\u20ac', '\u202c', '\u40fc',
	'\uf0b7', '\uf020', '\u25c9', '\u25c8', '\u25cd', '\u2207', '\uff89', '\ue056',
	'\uff9e', '\u200c', '\u25b5', '\uff62', '\uff63', '\u2217', '\uf0a3', '\u0e34',
	'\xab', '\xbb', '\u2044', '\uf052', '\u20bf', '\u25b4', '\xb9', '\xad', '\u03d6',
	'\u2011', '\u2666', '\xa5', '\u2006', '\u202d', '\u2005', '\uf236', '\uf525', 
	'�', '', '', '']

	for i in bad_str:
		text = text.replace(i, '')

	return text

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

def create_corpus(dir_name):
	max_len = 0
	total_sen = 0
	
	if 'chfinann' in dir_name:
		fout = open(os.path.join(dir_name, 'corpus.json'), 'w')
		fin = open(os.path.join(dir_name, 'train.json'))
		converter = opencc.OpenCC('t2s.json')
		sentence_tokenizer = zh_sen_tokenizer

		too_long, is_num_num = 0, 0
		corpus = []
		chifinann_datas = json.load(fin)
		for data in tqdm(chifinann_datas):  # data is ['id', {...}]
			sentences = data[1]['sentences']
			sentences = [converter.convert(s) for s in sentences]
			sentences = sentence_tokenizer(''.join(sentences))
			if type(sentences[0]) is str:
				for sentence in sentences:
					total_sen += 1
					if len(sentence) < 256:
						if max_len < len(sentence): max_len = len(sentence)
						is_num = 0
						for s in sentence:
							if s.isnumeric(): is_num += 1
						if is_num / len(sentence) > 0.25: is_num_num += 1
						else: corpus.append(json.dumps({'text': text_clean(sentence)}, ensure_ascii=False))
					else: too_long += 1
		fout.write('\n'.join(corpus))
		print(f'{dir_name} Done!\tmax_len: {max_len}\ttotal_sen: {total_sen}\ttoo_long: {too_long}\tis_num_num: {is_num_num}')
	
	elif 'covid19' in dir_name or 'pandemic' in dir_name:
		fout = open(os.path.join(dir_name, 'corpus.json'), 'w')
		fin = open(os.path.join(dir_name, 'corpus.txt'))
		sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		too_long, is_num_num = 0, 0
		corpus = []
		lines = fin.readlines()
		for line in tqdm(lines):
			line = text_clean(line.strip())
			sentences = sentence_tokenizer.tokenize(line)
			for sentence in sentences:
				total_sen += 1
				if len(sentence.split()) < 256:
					if max_len < len(sentence.split()): max_len = len(sentence.split())
					is_num = 0
					for s in sentence:
						if s.isnumeric(): is_num += 1
					if is_num / len(sentence) > 0.25: is_num_num += 1
					else: corpus.append(json.dumps({'text': text_clean(sentence)}, ensure_ascii=False))
				else: too_long += 1
		fout.write('\n'.join(corpus))
		print(f'{dir_name} Done!\tmax_len: {max_len}\ttotal_sen: {total_sen}\ttoo_long: {too_long}\tis_num_num: {is_num_num}')

	elif 'ere' in dir_name:
		fout = open(os.path.join(dir_name, 'corpus.json'), 'w')
		fin = open(os.path.join(dir_name, 'train.oneie.json'))

		corpus = []
		lines = fin.readlines()
		for line in tqdm(lines):
			data = json.loads(line)
			total_sen += 1
			if max_len < len(data['tokens']): max_len = len(data['tokens'])
			corpus.append(json.dumps({'text': ' '.join(data['tokens'])}, ensure_ascii=False))
		fout.write('\n'.join(corpus))
		print(f'{dir_name} Done!\tmax_len: {max_len}\ttotal_sen: {total_sen}')

	elif 'nyt' in dir_name:
		fout1 = open(os.path.join(dir_name, 'corpus.5k.json'), 'w')
		fout2 = open(os.path.join(dir_name, 'corpus.json'), 'w')
		all_file_path = []
		input_dir = os.path.join(dir_name, 'LDC2008T19', 'data')
		for year in os.listdir(input_dir):
			for dir_id in os.listdir(input_dir + '/' + year):
				if 'tgz' in dir_id: continue
				for dir_deep_id in os.listdir(input_dir + '/' + year + '/' + dir_id):
					for xml_id in os.listdir(input_dir + '/' + year + '/' + dir_id + '/' + dir_deep_id):
						all_file_path.append(input_dir + '/' + year + '/' + dir_id + '/' + dir_deep_id + '/' + xml_id)
		sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		too_long, is_num_num = 0, 0
		corpus = []
		for xml_file_path in tqdm(all_file_path):
			sentences = []
			xml_file = open(xml_file_path)
			et = parse(xml_file)
			root = et.getroot()
			for e in root.iter():
				if e.tag == 'title' or e.tag == 'hl1':
					if e.text != None: sentences.append(e.text.strip() + '.')
				elif e.tag == 'p':
					if e.text != None: sentences.append(e.text.strip())
			sentences = sentence_tokenizer.tokenize(' '.join(sentences))  # is still a list
			for sentence in sentences:
				total_sen += 1
				if len(sentence.split()) < 256:
					if max_len < len(sentence.split()): max_len = len(sentence.split())
					is_num = 0
					for s in sentence:
						if s.isnumeric(): is_num += 1
					if is_num / len(sentence) > 0.25: is_num_num += 1
					else: corpus.append(json.dumps({'text': text_clean(sentence)}, ensure_ascii=False))
				else: too_long += 1
		fout1.write('\n'.join(corpus[:5000]))
		fout2.write('\n'.join(corpus))
		print(f'{dir_name} Done!\tmax_len: {max_len}\ttotal_sen: {total_sen}\ttoo_long: {too_long}\tis_num_num: {is_num_num}')

	elif 'rmrb' in dir_name:
		fout1 = open(os.path.join(dir_name, 'corpus.5k.json'), 'w')
		fout2 = open(os.path.join(dir_name, 'corpus.json'), 'w')
		fin = open(os.path.join(dir_name, 'rmrb-1946-2001.json'))
		converter = opencc.OpenCC('t2s.json')
		sentence_tokenizer = zh_sen_tokenizer

		too_long, is_num_num = 0, 0
		corpus = []
		lines = fin.readlines()
		for line in tqdm(lines):
			data = json.loads(line)  # dict_keys(['_id', 'url', 'date', 'content', 'title'])
			raw_text = data['content'].strip()
			raw_text = text_clean(raw_text)
			sentences = raw_text.split()[1:] # the first sentence is "第四栏云云"
			if len(sentences) == 0: continue
			sentences = [converter.convert(s) for s in sentences]
			sentences = sentence_tokenizer(''.join(sentences))
			if type(sentences[0]) is str:
				for sentence in sentences:
					total_sen += 1
					if len(sentence) < 256:
						if max_len < len(sentence): max_len = len(sentence)
						is_num = 0
						for s in sentence:
							if s.isnumeric(): is_num += 1
						if is_num / len(sentence) > 0.25: is_num_num += 1
						else: corpus.append(json.dumps({'text': text_clean(sentence)}, ensure_ascii=False))
					else: too_long += 1
		fout1.write('\n'.join(corpus[:5000]))
		fout2.write('\n'.join(corpus))
		print(f'{dir_name} Done!\tmax_len: {max_len}\ttotal_sen: {total_sen}\ttoo_long: {too_long}\tis_num_num: {is_num_num}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--chfinann', default='data/chfinann')
	parser.add_argument('--covid19', default='data/covid19')
	parser.add_argument('--ere', default='data/ere')
	parser.add_argument('--nyt', default='data/nyt')
	parser.add_argument('--pandemic', default='data/pandemic')
	parser.add_argument('--rmrb', default='data/rmrb')
	args = parser.parse_args()
	print(vars(args))

	create_corpus(args.chfinann)
	create_corpus(args.covid19)
	create_corpus(args.ere)
	create_corpus(args.nyt)
	create_corpus(args.pandemic)
	create_corpus(args.rmrb)

	print('Done')