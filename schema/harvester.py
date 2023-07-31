# coding=utf-8

import re
import os
import json
import argparse
from tqdm import tqdm
import prettytable as pt
import random
import math
import numpy as np
import collections
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import wordnet as wn
import OpenHowNet

from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from base.evaluater import check_with_bcubed_lib

class Vertex():
	def __init__(self, vid, cid, nodes, k_in=0):
		self._vid = vid
		self._cid = cid
		self._nodes = nodes
		self._kin = k_in

class Louvain():
	def __init__(self, G):
		self._G = G
		self._m = 0
		self._cid_vertices = {}
		self._vid_vertex = {}
		for vid in self._G.keys():
			self._cid_vertices[vid] = set([vid]) 
			self._vid_vertex[vid] = Vertex(vid, vid, set([vid]))
			self._m += sum([1 for neighbor in self._G[vid].keys() if neighbor > vid])

	def first_stage(self):
		mod_inc = False
		visit_sequence = self._G.keys()
		random.shuffle(list(visit_sequence))
		while True:
			can_stop = True
			for v_vid in visit_sequence:
				v_cid = self._vid_vertex[v_vid]._cid
				k_v = sum(self._G[v_vid].values()) + self._vid_vertex[v_vid]._kin
				cid_Q = {}
				for w_vid in self._G[v_vid].keys():
					w_cid = self._vid_vertex[w_vid]._cid
					if w_cid in cid_Q: continue
					else:
						tot = sum([sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
						if w_cid == v_cid: tot -= k_v
						k_v_in = sum([v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
						delta_Q = k_v_in - k_v * tot / self._m
						cid_Q[w_cid] = delta_Q
				cid, max_delta_Q = sorted(cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
				if max_delta_Q > 0.0 and cid != v_cid:
					self._vid_vertex[v_vid]._cid = cid
					self._cid_vertices[cid].add(v_vid)
					self._cid_vertices[v_cid].remove(v_vid)
					can_stop = False
					mod_inc = True
			if can_stop: break
		return mod_inc 

	def second_stage(self):
		cid_vertices = {}
		vid_vertex = {}
		for cid, vertices in self._cid_vertices.items():
			if len(vertices) == 0: continue
			new_vertex = Vertex(cid, cid, set()) 
			for vid in vertices:
				new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
				new_vertex._kin += self._vid_vertex[vid]._kin
				for k, v in self._G[vid].items():
					if k in vertices:
						new_vertex._kin += v / 2.0
			cid_vertices[cid] = set([cid])
			vid_vertex[cid] = new_vertex

		G = collections.defaultdict(dict)
		for cid1, vertices1 in self._cid_vertices.items():
			if len(vertices1) == 0: continue
			for cid2, vertices2 in self._cid_vertices.items():
				if cid2 <= cid1 or len(vertices2) == 0: continue
				edge_weight = 0.0
				for vid in vertices1:
					for wid, v in self._G[vid].items():
						if wid in vertices2:
							edge_weight += v
				if edge_weight != 0:
					G[cid1][cid2] = edge_weight
					G[cid2][cid1] = edge_weight
		self._cid_vertices = cid_vertices
		self._vid_vertex = vid_vertex
		self._G = G

	def get_communities(self):
		communities = []
		for vertices in self._cid_vertices.values():
			if len(vertices) != 0:
				c = set()
				for vid in vertices:
					c.update(self._vid_vertex[vid]._nodes)
				communities.append(c)
		return communities

	def execute(self):
		iter_time = 1
		while True:
			print ('Louvain epoch:\t', iter_time)
			iter_time += 1
			mod_inc = self.first_stage()
			if mod_inc:
				self.second_stage()
			else: break
		return self.get_communities()

def PageRank(G, R, T=300, eps=1e-6, beta=0.8):
	 # normalize
	 N = R.shape[0]
	 M = np.zeros((N, N))
	 for i in range(N):
		 D_i = sum(G[i])
		 if D_i == 0: continue
		 for j in range(N):
			 M[j][i] = G[i][j] / D_i # watch out! M_j_i instead of M_i_j
	 R = R / sum(R)

	 # power iter
	 teleport = np.ones(N) / N
	 for time in range(T):
		 R_new = beta * np.dot(M, R) + (1 - beta) * teleport
		 if np.linalg.norm(R_new - R) < eps:
			 break
		 R = R_new.copy()

	 return R_new

def WordnetSim(word1, word2):
	path_sim = 0
	synsets1 = wn.synsets(word1)
	synsets2 = wn.synsets(word2)
	for synset1 in synsets1:
		for synset2 in synsets2:
			sim = synset1.path_similarity(synset2)
			if sim is not None:
				path_sim = max(path_sim, sim)
	return path_sim

def HownetSim(word1, word2):
	 return hownet_dict_advanced.calculate_word_similarity(word1, word2)

def BertSim(word1, word2):
	vec1 = bert_model.encode(word1)
	vec2 = bert_model.encode(word2)
	return cosine_similarity([vec1], [vec2])[0][0]

def similarity(str1, str2, mode):
	if mode == 'bow':
		return int(sorted(str1) == sorted(str2))
	elif mode == 'lexical':
		count = 0
		for w1 in str1:
			for w2 in str2:
				if w1 == w2: count += 1
		return float(count) / (len(str1) * len(str2))

def interCluster(lm_outputs, mode, verb_lambda=3, type_lambda=1):
	G = collections.defaultdict(dict)
	events = []
	vos = []
	for i, instance1 in enumerate(tqdm(lm_outputs)):
		vo1 = instance1[0]
		output1 = instance1[1]
		event1 = output1[0]
		events.append(event1)
		vos.append(vo1)
		for j, instance2 in enumerate(lm_outputs):
			vo2 = instance2[0]
			output2 = instance2[1]
			event2 = output2[0]
			if i != j:
				e1 = vo1.lower().strip()
				e2 = vo2.lower().strip()
				e1 = [e1.split()[0]]
				e2 = [e2.split()[0]]
				sim_verb = similarity(e1, e2, 'bow') * verb_lambda
				if mode == 'VerbMatch':
					sim = sim_verb
				if mode == 'TypeMatch':
					e1 = event1.lower().strip()
					e2 = event2.lower().strip()
					if '/' in e1: e1 = e1.split('/')
					elif '-' in e1: e1 = e1.split('/')
					else: e1 = [e1]
					if '/' in e2: e2 = e2.split('/')
					elif '-' in e2: e2 = e2.split('/')
					else: e2 = [e2]
					if e1[0] == 'none' or e2[0] =='none': sim_type = 0
					else: sim_type = similarity(e1, e2, 'lexical')
					sim = sim_verb + sim_type * type_lambda
				G[i][j] = sim
				G[j][i] = sim
	algorithm = Louvain(G)
	communities = algorithm.execute()
	communities = sorted(communities, key=lambda b: -len(b))
	
	predicted = [0] * len(lm_outputs)
	all_c = 0
	for count, communitie in enumerate(communities):
		all_c += len(communitie)
		for c in communitie:
			predicted[c] = count
	results = [[vos[c] for c in communitie] for communitie in communities]
	assert all_c == len(lm_outputs)
	return predicted, results

def intraCluster(lm_outputs, mode, simfunc, tfidf_lambda, pagerank_lambda, score_threshold):
	total_num = len(lm_outputs)
	gloabl_slot_freq = {}
	gloabl_mat = {}
	for output in lm_outputs:
		vo = output[0]
		samples = output[1]
		all_slots = []
		for sample in samples:
			event, slots = sample[0], sample[1:]
			all_slots.extend(slots)
			for slot1 in slots:
				for slot2 in slots:
					if slot1 == slot2: continue
					else:
						key = '-'.join(sorted([slot1, slot2]))
						if key not in gloabl_mat: gloabl_mat[key] = 1
						else: gloabl_mat[key] += 1

		for slot in all_slots:
			if slot not in gloabl_slot_freq: gloabl_slot_freq[slot] = 1
			else: gloabl_slot_freq[slot] += 1
	
	new_inputs = []
	all_final_slots = []
	for output in tqdm(lm_outputs):
		vo = output[0]
		samples = output[1]
		all_slots = []
		for sample in samples:
			event, slots = sample[0], sample[1:]
			all_slots.extend(slots)

		local_slot_freq = {}
		for slot in all_slots:
			if slot not in local_slot_freq: local_slot_freq[slot] = 1
			else: local_slot_freq[slot] += 1
		all_slots = list(set(all_slots))

		tfidf_slots, pagerank_slots, net_slots, new_events = None, None, None, []
		sum_tfidf = 0
		if 'TF-IDF' in mode:
			tfidf_slots = []
			for slot in all_slots:
				tfidf = (1+math.log(local_slot_freq[slot], 10)**2) * math.log(total_num/gloabl_slot_freq[slot], 10)
				tfidf_slots.append((slot, round(tfidf, 4)))
				sum_tfidf += tfidf
			# tfidf_slots = sorted(tfidf_slots, key=lambda x: x[1], reverse=True)
			# print (tfidf_slots)

		if 'PageRank' in mode:
			node_num = len(all_slots)
			init_r = np.ones(node_num) / (node_num)
			local_mat = np.zeros((node_num, node_num))
			for i in range(node_num):
				for j in range(node_num):
					if i == j: continue
					else:
						key = '-'.join(sorted([all_slots[i], all_slots[j]]))
						if key in gloabl_mat: local_mat[i][j] = gloabl_mat[key]
			new_r = PageRank(local_mat, init_r)
			if sum_tfidf == 0: sum_tfidf = 1
			pagerank_slots = [(slot, round(pr * sum_tfidf, 4)) for slot, pr in zip(all_slots, new_r)]
			# pagerank_slots = sorted(pagerank_slots, key=lambda x: x[1], reverse=True)
			# print (pagerank_slots)

		if 'WordSense' in mode:
			verb = vo.split()[0]
			obj = vo.split()[1]
			net_slots = {slot: 0 for slot in all_slots}
			for sample in samples:
				event, slots = sample[0], sample[1:]
				if '/' in event:
					sim = 0
					new_event = None
					o_flag = False
					for tmpevent in event.split('/'):
						sim_v = simfunc(tmpevent, verb)
						sim_o = simfunc(tmpevent, obj)
						tmpsim = max(sim_v, sim_o)
						if tmpsim > sim:
							sim = tmpsim
							new_event = tmpevent
						if sim_o > sim_v: o_flag = True
					if new_event is not None and o_flag and new_event!=obj:
						new_event = new_event + '-' + obj
					elif o_flag and event!=obj:
						event = event + '-' + obj
					new_events.append(new_event) if new_event is not None else new_events.append(event)
				elif '-' in event:
					sim = 0
					new_event = None
					o_flag = False
					for tmpevent in event.split('-'):
						sim_v = simfunc(tmpevent, verb)
						sim_o = simfunc(tmpevent, obj)
						tmpsim = max(sim_v, sim_o)
						if tmpsim > sim:
							sim = tmpsim
							new_event = tmpevent
						if sim_o > sim_v: o_flag = True
					if new_event is not None and o_flag and new_event!=obj:
						new_event = new_event + '-' + obj
					elif o_flag and event!=obj:
						event = event + '-' + obj
					new_events.append(new_event) if new_event is not None else new_events.append(event)
				else:
					sim_v = simfunc(event, verb)
					sim_o = simfunc(event, obj)
					sim = max(sim_v, sim_o)
					if sim_o > sim_v and event!=obj:
						event = event + '-' + obj
					new_events.append(event)
				for slot in slots:
					if slot in all_slots:
						net_slots[slot] = max(net_slots[slot], sim)
			net_slots = [(slot, round(sim, 4)) for slot, sim in net_slots.items()]
			# net_slots = sorted(net_slots, key=lambda x: x[1], reverse=True)
			# print (net_slots)
			if sorted(net_slots, key=lambda x: x[1], reverse=True)[0][1] == 0: net_slots = None

		final_slots = []
		for i, slot in enumerate(all_slots):
			final_score = 0
			if tfidf_slots is not None:
				final_score += tfidf_slots[i][1] * tfidf_lambda
			if pagerank_slots is not None:
				final_score += pagerank_slots[i][1] * pagerank_lambda
			if net_slots is not None:
				final_score *= net_slots[i][1]
			final_slots.append((slot, final_score))
		if len(final_slots) > 0:
			final_scores = np.array([slot[1] for slot in final_slots])
			final_scores = (final_scores - min(final_scores)) / (max(final_scores) - min(final_scores))
			final_slots = [(slot[0], round(score, 4), round(slot[1], 4)) for slot, score in zip(final_slots, final_scores) if score >= score_threshold]
		final_slot_names = [slot[0] for slot in final_slots]

		if len(new_events) == 0:
			for sample in samples:
				event, slots = sample[0], sample[1:]
				new_events.append(event)

		final_events = []
		for event, sample in zip(new_events, samples):
			count = 0
			slots = sample[1:]
			for slot in slots:
				if slot in final_slot_names:
					count += final_slots[final_slot_names.index(slot)][1]
			final_events.append((event, count))
		final_events = sorted(final_events, key=lambda x: x[1], reverse=True)
		new_inputs.append((vo, [final_events[0][0]] + final_slot_names))
		all_final_slots.append((tfidf_slots, pagerank_slots, net_slots, final_slots, final_events[0][0]))

	return new_inputs, all_final_slots

def esher(lm_outputs, data_dict, id_map, output_dir, intra=None, inter=None, times=3, tfidf_lambda=1, pagerank_lambda=1, score_threshold=1/3):
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	golden = [id_map.index(v) for k, v in data_dict['event_type'].items()]
	random.seed(1234)
	np.random.seed(1234)
	
	if intra is None:
		all_ARI = []
		all_NMI = []
		all_ACC = []
		all_Bcubed_F1 = []
		all_clusters = []
		for time in range(times):
			predicted, results = interCluster([(output[0], output[1][time]) for output in lm_outputs], inter)

			ARI = adjusted_rand_score(golden, predicted)
			NMI = normalized_mutual_info_score(golden, predicted)
			ACC = 0
			Bcubed_F1 = check_with_bcubed_lib(golden, predicted)

			all_ARI.append(ARI)
			all_NMI.append(NMI)
			all_ACC.append(ACC)
			all_Bcubed_F1.append(Bcubed_F1)
			all_clusters.append(len(results))
			
			with open(os.path.join(output_dir, f'{time}.json'), 'w') as f:
				for i, cluster in enumerate(results):
					result_string = f"Topic {i} ({len(cluster)}): "
					for vo in cluster:
						result_string += ', ' + vo
					f.write(result_string + '\n')
				f.write('\n')
				tb = pt.PrettyTable()
				tb.field_names = ['ARI', 'NMI', 'ACC', 'BCubed-F1', 'V-measure']
				tb.add_row([round(ARI*100, 2), round(NMI*100, 2), round(ACC*100, 2), round(Bcubed_F1*100, 2), round(len(results), 2)])
				f.write(str(tb))
				
		return np.array(all_ARI).mean(), np.array(all_NMI).mean(), np.array(all_ACC).mean(), np.array(all_Bcubed_F1).mean(), np.array(all_clusters).mean(), \
			np.array(all_ARI).std(), np.array(all_NMI).std(), np.array(all_ACC).std(), np.array(all_Bcubed_F1).std(), np.array(all_clusters).std()		
	else:
		new_inputs, _ = intraCluster(lm_outputs, intra, WordnetSim, tfidf_lambda, pagerank_lambda, score_threshold)
		predicted, results = interCluster(new_inputs, inter)
		ARI = adjusted_rand_score(golden, predicted)
		NMI = normalized_mutual_info_score(golden, predicted)
		ACC = 0
		Bcubed_F1 = check_with_bcubed_lib(golden, predicted)

		with open(os.path.join(output_dir, '0.json'), 'w') as f:
			for i, cluster in enumerate(results):
				result_string = f"Topic {i} ({len(cluster)}): "
				for vo in cluster:
					result_string += ', ' + vo
				f.write(result_string + '\n')
			f.write('\n')
			tb = pt.PrettyTable()
			tb.field_names = ['ARI', 'NMI', 'ACC', 'BCubed-F1', 'V-measure']
			tb.add_row([round(ARI*100, 2), round(NMI*100, 2), round(ACC*100, 2), round(Bcubed_F1*100, 2), round(len(results), 2)])
			f.write(str(tb))
		
		return ARI, NMI, ACC, Bcubed_F1, len(results), 0, 0, 0, 0, 0

def get_lm_outputs(input_file, sample_num, incontext_index, salient_verbs, salient_obj_heads):
	with open(input_file, encoding='utf-8') as fin:
		results = ''.join(fin.readlines())
		results = results.split('\n\n###\n\n')[:-1]

		instance = []
		for result in results:
			result = result.split('\n')[incontext_index]
			vo = result.split('&')[0].strip()
			verb_lemma = vo.split()[0]
			obj_head_lemma = vo.split()[1]
			if verb_lemma not in salient_verbs or obj_head_lemma not in salient_obj_heads:
				continue

			result = result.split('&')[1].strip().split()

			if len(result) == 0: 
				instance.append(['none', 'none'])
			elif len(result) == 1: 
				instance.append([result[0], 'none'])
			else: 
				instance.append([result[0]] + list(set(result[1:])))
		
			if len(instance) == sample_num:
				yield (vo, instance, salient_verbs[verb_lemma]*salient_obj_heads[obj_head_lemma])
				instance = []

def agg_results(lm_outputs, output_file, annotate_file, simfunc, tfidf_lambda, pagerank_lambda, score_threshold, slot_cluster_lambda, link_sentence, corpus_parsed_svo, salient_verbs, salient_obj_heads, obj2obj_head_infos):
	if link_sentence:
		link2sentences = {}
		for k, v in tqdm(corpus_parsed_svo.items()):
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
								key = ' '.join([verb_lemma, obj_head_lemma])
								if key not in link2sentences: 
									link2sentences[key] = [v['raw_sentence']]
								else:
									link2sentences[key].append(v['raw_sentence'])

	with open(output_file, 'w', encoding='utf-8') as fout:
		#########################################   Intra Instance Aggregation   ###########################################

		# new_inputs, all_final_slots = intraCluster(lm_outputs, ['TF-IDF', 'PageRank', 'WordSense'], simfunc, tfidf_lambda, pagerank_lambda, score_threshold)
		# new_inputs, all_final_slots = intraCluster(lm_outputs, ['TF-IDF', 'WordSense'], simfunc, tfidf_lambda, pagerank_lambda, score_threshold)
		new_inputs, all_final_slots = intraCluster(lm_outputs, ['TF-IDF', 'PageRank'], simfunc, tfidf_lambda, pagerank_lambda, score_threshold)
		# new_inputs, all_final_slots = intraCluster(lm_outputs, ['PageRank', 'WordSense'], simfunc, tfidf_lambda, pagerank_lambda, score_threshold)

		total_tb = {}
		for instance, f_slots, n_inputs in zip(lm_outputs, all_final_slots, new_inputs):
			vo = instance[0]
			samples = instance[1]
			tfidf_slots, pagerank_slots, net_slots, final_slots, new_event = f_slots
			if tfidf_slots is not None: tfidf_slots = sorted(tfidf_slots, key=lambda x: x[1], reverse=True)
			if pagerank_slots is not None: pagerank_slots = sorted(pagerank_slots, key=lambda x: x[1], reverse=True)
			if net_slots is not None: net_slots = sorted(net_slots, key=lambda x: x[1], reverse=True)
			if final_slots is not None: final_slots = sorted(final_slots, key=lambda x: x[1], reverse=True)

			max_num_slots = 0
			for sample in samples:
				if len(sample[1:]) > max_num_slots: max_num_slots = len(sample)
			if tfidf_slots is not None and len(tfidf_slots) > max_num_slots: max_num_slots = len(tfidf_slots)
			if pagerank_slots is not None and len(pagerank_slots) > max_num_slots: max_num_slots = len(pagerank_slots)
			if net_slots is not None and len(net_slots) > max_num_slots: max_num_slots = len(net_slots)
			if final_slots is not None and len(final_slots) > max_num_slots: max_num_slots = len(final_slots)

			for sample in samples:
				for i in range(len(sample[1:]), max_num_slots): sample.append('/')
			if tfidf_slots is not None: 
				for i in range(len(tfidf_slots), max_num_slots): tfidf_slots.append('/')
			else: tfidf_slots = ['/'] * max_num_slots
			if pagerank_slots is not None: 
				for i in range(len(pagerank_slots), max_num_slots): pagerank_slots.append('/')
			else: pagerank_slots = ['/'] * max_num_slots
			if net_slots is not None: 
				for i in range(len(net_slots), max_num_slots): net_slots.append('/')
			else: net_slots = ['/'] * max_num_slots
			if final_slots is not None: 
				for i in range(len(final_slots), max_num_slots): final_slots.append('/')
			else: final_slots = ['/'] * max_num_slots

			tb = pt.PrettyTable()
			tb.field_names = [vo] + [f'Sample{i+1}' for i in range(len(samples))] + ['TF-IDF Score', 'PageRank Score', 'WordSense Score', 'Final Results']
			tb.add_row(['Event Type'] + [sample[0].upper() for sample in samples] + ['?'] + ['?'] + ['?'] + [new_event.upper()])
			for i in range(max_num_slots):
				tb.add_row([f'Slot{i+1}'] + [sample[i+1] for sample in samples] + [tfidf_slots[i], pagerank_slots[i], net_slots[i], final_slots[i]])
			
			verb, obj = vo.split()[0], vo.split()[1]
			if link_sentence:
				sentences = list(set(link2sentences[vo]))
				total_tb[vo] = (salient_verbs[verb], salient_obj_heads[obj], tb, n_inputs, sentences)
			else:
				total_tb[vo] = (salient_verbs[verb], salient_obj_heads[obj], tb, n_inputs)

		#########################################   Inter Instance Aggregation   ###########################################

		clustered_outputs = []
		_, results = interCluster(new_inputs, 'TypeMatch')
		for i, cluster in enumerate(results):
			clustered_vo = []
			samples = []
			for vo in cluster:
				clustered_vo.append(vo)
				samples.append(total_tb[vo][3][1])
			clustered_outputs.append((clustered_vo, samples))
		
		#########################################   Slots Clustering   ###########################################

		_, all_final_slots = intraCluster(clustered_outputs, ['TF-IDF', 'PageRank'], simfunc, tfidf_lambda, pagerank_lambda, 0)

		all_final_tbs = []
		for num_cluster, cluster in enumerate(results):
			tmp_total_tb = []
			for vo in cluster:
				tmp_total_tb.append(total_tb[vo])
			tmp_total_tb = sorted(tmp_total_tb, key=lambda x: (x[0]*x[1]), reverse=True)

			if link_sentence:
				for (s1, s2, tb, _, sens) in tqdm(tmp_total_tb):
					for i, sen in enumerate(sens):
						fout.write(str(i) + ':\t' + sen + '\n')
					fout.write(str(s1*s2) + '\t' + str(len(sens)) + '\t' + str(s1) + '\t' + str(s2) + '\n')
					fout.write(str(tb) + '\n')
			else:
				for (s1, s2, tb, _) in tqdm(tmp_total_tb):
					fout.write(str(s1*s2) + '\t' + str(s1) + '\t' + str(s2) + '\n')
					fout.write(str(tb) + '\n')
			
			tfidf_slots, pagerank_slots, _, final_slots, new_event = all_final_slots[num_cluster]
		
			for vo in cluster:
				fout.write(str(vo) + '\t')
			fout.write('\n')

			while True:
				G = collections.defaultdict(dict)
				for i, slot1 in enumerate(final_slots):
					slot1 = slot1[0]
					for j, slot2 in enumerate(final_slots):
						slot2 = slot2[0]
						if i != j:
							# sim = (simfunc(slot1, slot2) + BertSim(slot1, slot2)) / 2 * slot_cluster_lambda
							sim = BertSim(slot1, slot2) * slot_cluster_lambda
							G[i][j] = sim
							G[j][i] = sim
				random.seed(1234)
				np.random.seed(1234)
				algorithm = Louvain(G)
				communities = algorithm.execute()
				communities = sorted(communities, key=lambda b: -len(b))
				if len(communities) == 0 or len(communities[0]) == 1: break
				elif len(communities[0]) > len(final_slots)/2 and slot_cluster_lambda<5: slot_cluster_lambda = slot_cluster_lambda + 0.5
				else: break

			visited_slots = []
			clustered_slots = []
			for communitie in communities:
				tmp_slots = []
				tmp_slot_score = 0
				for c in communitie:
					visited_slots.append(final_slots[c][0])
					tmp_slots.append((final_slots[c][0], final_slots[c][1], final_slots[c][2], tfidf_slots[c][1], pagerank_slots[c][1]))
					tmp_slot_score += final_slots[c][1]
					# tmp_slot_score = max(final_slots[c][1], tmp_slot_score)
					fout.write(str((final_slots[c][0], final_slots[c][1], final_slots[c][2], tfidf_slots[c][1], pagerank_slots[c][1])) + '\t')
				tmp_slot_name = sorted(tmp_slots, key=lambda x: x[1], reverse=True)[0][0]
				clustered_slots.append((tmp_slot_name, tmp_slot_score))
				fout.write('\n')
			
			# for slot in final_slots:
			# 	if slot[0] not in visited_slots:
			# 		clustered_slots.append((slot[0], slot[1]))

			if tfidf_slots is not None: tfidf_slots = sorted(tfidf_slots, key=lambda x: x[1], reverse=True)
			if pagerank_slots is not None: pagerank_slots = sorted(pagerank_slots, key=lambda x: x[1], reverse=True)
			if final_slots is not None: final_slots = sorted(final_slots, key=lambda x: x[1], reverse=True)
			if len(clustered_slots) > 0:
				clustered_scores = np.array([slot[1] for slot in clustered_slots])
				# clustered_scores = (clustered_scores - min(clustered_scores)) / (max(clustered_scores) - min(clustered_scores))
				clustered_slots = [(slot[0], round(score, 4), round(slot[1], 4)) for slot, score in zip(clustered_slots, clustered_scores) if score > score_threshold]
				clustered_slots = sorted(clustered_slots, key=lambda x: x[1], reverse=True)
			else: clustered_slots = final_slots
			clustered_slots_names = [slot[0] for slot in clustered_slots]

			clustered_events = []
			clustered_vo, samples = clustered_outputs[num_cluster]
			for vo, sample in zip(clustered_vo, samples):
				count = 0
				event = sample[0]
				slots = sample[1:]
				for slot in slots:
					if slot in clustered_slots_names:
						count += clustered_slots[clustered_slots_names.index(slot)][1] * total_tb[vo][0] * total_tb[vo][1]
				clustered_events.append((event, count))
			clustered_events = sorted(clustered_events, key=lambda x: x[1], reverse=True)
			
			#########################################   Display   ###########################################

			max_num_slots = 0
			if tfidf_slots is not None and len(tfidf_slots) > max_num_slots: max_num_slots = len(tfidf_slots)
			if pagerank_slots is not None and len(pagerank_slots) > max_num_slots: max_num_slots = len(pagerank_slots)
			if final_slots is not None and len(final_slots) > max_num_slots: max_num_slots = len(final_slots)

			if tfidf_slots is not None: 
				for i in range(len(tfidf_slots), max_num_slots): tfidf_slots.append('/')
			else: tfidf_slots = ['/'] * max_num_slots
			if pagerank_slots is not None: 
				for i in range(len(pagerank_slots), max_num_slots): pagerank_slots.append('/')
			else: pagerank_slots = ['/'] * max_num_slots
			if final_slots is not None: 
				for i in range(len(final_slots), max_num_slots): final_slots.append('/')
			else: final_slots = ['/'] * max_num_slots
			if clustered_slots is not None: 
				for i in range(len(clustered_slots), max_num_slots): clustered_slots.append('/')
			else: clustered_slots = ['/'] * max_num_slots

			tb = pt.PrettyTable()
			tb.field_names = ['TF-IDF Score', 'PageRank Score', 'Final Results', 'Clustered Results']
			tb.add_row(['?'] + ['?'] + [new_event.upper()] + [clustered_events[0][0].upper()])
			for i in range(max_num_slots):
				tb.add_row([tfidf_slots[i], pagerank_slots[i], final_slots[i], clustered_slots[i]])
			fout.write('###########################################\n')
			fout.write(str(tb) + '\n')
			fout.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')

			all_final_tbs.append((len(cluster), cluster, tb))

	with open(annotate_file, 'w', encoding='utf-8') as fann:
		need_ann = {'intra': {}, 'final': {}}
		tbs = []
		for k, v in total_tb.items():
			tbs.append((k, v))
		tbs = sorted(tbs, key=lambda x: (x[1][0]*x[1][1]), reverse=True)
		for i, tb in enumerate(tbs):
			vo = tb[0]
			score, this_tb = tb[1][0] * tb[1][1], tb[1][2]
			this_tb.border = False
			this_tb.header = False
			sentences = tb[1][4] if link_sentence else None
			all_slots = [x.strip() for x in this_tb.get_string(fields=['Sample1']).split('\n') if x.strip() != '/'][1:] + [x.strip() for x in this_tb.get_string(fields=['Sample2']).split('\n') if x.strip() != '/'][1:] + [x.strip() for x in this_tb.get_string(fields=['Sample3']).split('\n') if x.strip() != '/'][1:]
			pred_slots = [x.strip() for x in this_tb.get_string(fields=['Final Results']).split('\n') if x.strip() != '/']
			pred_type = pred_slots[0]
			pred_slots = [re.findall(r'\'(.+?)\'', x)[0] for x in pred_slots[1:]]
			all_slots = sorted(list(set(all_slots)))
			pred_slots = sorted(pred_slots)
			need_ann['intra'][i] = {
				'score': score,
				'sentences': sentences,
				'input': vo,
				'pred_type': pred_type,
				'all_slots': all_slots,
				'pred_slots': pred_slots,
				'ann_type': (0, None),
				'ann_slots': [],
			}
		for i, tb in enumerate(all_final_tbs):
			score, vos, this_tb = tb[0], tb[1], tb[2]
			this_tb.border = False
			this_tb.header = False
			all_slots = [x.strip() for x in this_tb.get_string(fields=['Final Results']).split('\n') if x.strip() != '/']
			all_slots = [re.findall(r'\'(.+?)\'', x)[0] for x in all_slots[1:]]
			pred_slots = [x.strip() for x in this_tb.get_string(fields=['Clustered Results']).split('\n') if x.strip() != '/']
			pred_type = pred_slots[0]
			pred_slots = [re.findall(r'\'(.+?)\'', x)[0] for x in pred_slots[1:]]
			all_slots = sorted(list(set(all_slots)))
			pred_slots = sorted(pred_slots)
			need_ann['final'][i] = {
				'score': score,
				'input': vos,
				'pred_type': pred_type,
				'all_slots': all_slots,
				'pred_slots': pred_slots,
				'ann_type': (0, None),
				'ann_slots': []
			}
		json.dump(need_ann, fann, ensure_ascii=False, indent=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--language', default='en', choices=['en', 'zh'], help='english or chinese')
	parser.add_argument('--input_file', help='input lm file')
	parser.add_argument('--sample_num', default=3, help='the number of sample for each instance')
	parser.add_argument('--output_file', help='output result file')
	parser.add_argument('--annotate_file', help='annotate result file')
	parser.add_argument('--data_file', help='original data file')
	parser.add_argument('--tfidf_lambda', default=1)
	parser.add_argument('--pagerank_lambda', default=1)
	parser.add_argument('--score_threshold', default=1/3)
	parser.add_argument('--slot_cluster_lambda', default=0.5)
	parser.add_argument('--link_sentence', action='store_true')
	parser.add_argument('--top_num', type=int, default=500)

	args = parser.parse_args()
	print(vars(args))

	if args.language == 'en':
		with open('data/ace05/incontext-0.json') as f_incontext:
			incontexts = json.load(f_incontext)
			incontext_index = len(incontexts)
		simfunc = WordnetSim
	elif args.language == 'zh':
		with open('data/duee/incontext-0.json') as f_incontext:
			incontexts = json.load(f_incontext)
			incontext_index = len(incontexts)
		simfunc = HownetSim
		print('Loading Hownet model')
		global hownet_dict_advanced 
		hownet_dict_advanced= OpenHowNet.HowNetDict(init_sim=True)

	global bert_model
	bert_model = SentenceTransformer('resources/distiluse-base-multilingual-cased-v2')

	print('Loading corpus and salient term files')
	with open(os.path.join(args.data_file, 'corpus_parsed_svo.json')) as f:
		corpus_parsed_svo = json.load(f)
	with open(os.path.join(args.data_file, 'corpus_parsed_svo_salient_verbs.json')) as f:
		salient_verbs = json.load(f)
	with open(os.path.join(args.data_file, 'corpus_parsed_svo_salient_obj_heads.json')) as f:
		salient_obj_heads = json.load(f) 
	with open(os.path.join(args.data_file, 'corpus_parsed_svo_obj2obj_head_info.json')) as f:
		obj2obj_head_infos = json.load(f)   

	lm_outputs = [output for output in get_lm_outputs(args.input_file, args.sample_num, incontext_index, salient_verbs, salient_obj_heads)]
	lm_outputs = sorted(lm_outputs, key=lambda x: x[2], reverse=True)[:args.top_num]
	lm_outputs = [(output[0], output[1]) for output in lm_outputs]
	agg_results(lm_outputs, args.output_file, args.annotate_file, simfunc, args.tfidf_lambda, args.pagerank_lambda, args.score_threshold, args.slot_cluster_lambda, args.link_sentence, corpus_parsed_svo, salient_verbs, salient_obj_heads, obj2obj_head_infos)

	print('Done!')
