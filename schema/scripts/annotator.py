import os
import argparse
import json
from colorama import Fore, init
init()

def annotate(filename, schema_filename):
    if schema_filename is not None:
        with open(schema_filename) as fin:
            schema = json.load(fin)
            schema_types = {'0': 'Not Match'}
            for i, etype in enumerate(schema.keys()):
                schema_types[str(i+1)] = etype
    else: schema = None
    with open(filename) as fin:
        data = json.load(fin)
    
    while input('--- Press Enter to Begin... ---\n') != '':
        print ('Please Press Enter...')
    os.system("clear")
    while True:
        select1 = input('--- Annotate \'intra\'(0) or \'final\'(1)? Press 0 or 1... ---\n') 
        if select1 not in ['0', '1']:
            print ('Please Press 0 or 1...')
        else: break
    
    if select1 == '0': target = 'intra'
    elif select1 == '1': target = 'final'

    for i, instance in data[target].items():
        if int(i) >= 100 and target == 'intra': continue
        instance['ann_type'] = (0, None)
        instance['ann_slots'] = []
        this_id = str(i)
        this_input = str(instance['input'])
        this_pred_type = str(instance['pred_type'])
        this_recall_slots = str(instance['all_slots'])
        this_pred_slots = str(instance['pred_slots'])
        os.system("clear")
        print (f'--- Annotating {target} of {filename}... ---')
        print (f'Instance {this_id}')
        if 'sentences' in instance.keys():
            for sentence in instance['sentences'][:3]:
                print (f'Sentence: {sentence}')
        print (f'Input: {this_input}')
        print (Fore.BLUE + f'Recall Slots: {this_recall_slots}' + Fore.BLACK)
        print (Fore.RED + f'Pred Event Type: {this_pred_type}' + Fore.BLACK)
        print (Fore.RED + f'Pred Slots: {this_pred_slots}' + Fore.BLACK)
        while True:
            select2 = input('--- The pred event type' + Fore.RED + f' {this_pred_type} ' + Fore.BLACK + 'is Accepted(1), Rejected(2) or Well but not related to this corpus(3)? ---\n') 
            if select2 not in ['1', '2', '3']:
                print ('Please Press 1, 2 or 3...')
            else: break
        if select2 == '1' and schema is not None:
            while True:
                select3 = input('--- If Accepted, does it match human annotated types or not? ---\n' + Fore.YELLOW + f' {str(schema_types)}' + Fore.BLACK + '\n' )
                if select3 not in schema_types.keys():
                    print ('Please input the id of the matched type')
                else: break
        else: select3 = '0'
        if select3 == '0': instance['ann_type'] = [select2, None]
        else: instance['ann_type'] = [select2, schema_types[select3]]

        for slot in instance['pred_slots']:
            while True:
                select4 = input('--- The pred event slot' + Fore.RED + f' {slot} ' + Fore.BLACK + 'is Accepted(1) or Rejected(2)? ---\n') 
                if select4 not in ['1', '2']:
                    print ('Please Press 1 or 2...')
                else: break
            if select4 == '1' and instance['ann_type'][1] is not None and instance['ann_type'][1] in schema.keys():
                tmp_slots = {'0': 'Not Match'}
                for i, s in enumerate(schema[instance["ann_type"][1]]):
                    tmp_slots[str(i+1)] = s
                while True:
                    select5 = input('--- If Accepted, does it match human annotated slots or not? ---\n' + Fore.YELLOW + f' {str(tmp_slots)}' + Fore.BLACK + '\n' )
                    if select5 not in tmp_slots.keys():
                        print ('Please input the id of the matched slot')
                    else: break
            else: select5 = '0'
            if select5 == '0': instance['ann_slots'].append((slot, select4, None))
            else: instance['ann_slots'].append((slot, select4, tmp_slots[select5]))

        while True:
            select6 = input('--- Does pred event slots need append from Recall(1) or Not(2)? ---\n') 
            if select6 not in ['1', '2']:
                print ('Please Press 1 or 2...')
            else: break
        instance['ann_slots'].append(('Recall', select6, None))

        while True:
            select7 = input('--- Does pred event slots need append from Human(1) or Not(2)? ---\n') 
            if select7 not in ['1', '2']:
                print ('Please Press 1 or 2...')
            else: break
        instance['ann_slots'].append(('Human', select7, None))

    with open(filename) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=True)

def evaluate(filename, schema_filename):
    if schema_filename is not None:
        with open(schema_filename) as fin:
            schema = json.load(fin)
    else: schema = None
    with open(filename) as fin:
        data = json.load(fin)

    humanslot = 0
    for _, s in schema.items():
        humanslot += len(s)

    disco1 = 0
    matched_type1 = []
    accept1 = 0
    slotnum1 = 0
    slotmatched1 = {k: [] for k, v in schema.items()}
    slotaccept1 = 0
    recall1 = 0
    for i, instance in data['intra'].items():
        if int(i) >= 100: continue
        disco1 += 1
        if instance['ann_type'][1] is not None:
            matched_type1.append(instance['ann_type'][1])
        if instance['ann_type'][0] == '1': accept1 += 1
        slotnum1 += len(instance['pred_slots'])
        for slot in instance['ann_slots']:
            if slot[0] == 'Recall':
                if slot[1] == '1': recall1 += 1
            elif slot[0] == 'Human': continue
            else:
                if slot[2] is not None:
                    slotmatched1[instance['ann_type'][1]].append(slot[2])
                if slot[1] == '1': slotaccept1 += 1

    smatched1 = 0
    for k, v in slotmatched1.items():
        smatched1 += len(set(v))

    matched_type2 = []
    accept2 = 0
    slotnum2 = 0
    slotmatched2 = {k: [] for k, v in schema.items()}
    slotaccept2 = 0
    recall2 = 0
    for i, instance in data['final'].items():
        if instance['ann_type'][1] is not None:
            matched_type2.append(instance['ann_type'][1])
        if instance['ann_type'][0] == '1': accept2 += 1
        slotnum2 += len(instance['pred_slots'])
        for slot in instance['ann_slots']:
            if slot[0] == 'Recall': 
                if slot[1] == '1': recall2 += 1
            elif slot[0] == 'Human': continue
            else:
                if slot[2] is not None:
                    slotmatched2[instance['ann_type'][1]].append(slot[2])
                if slot[1] == '1': slotaccept2 += 1

    smatched2 = 0
    for k, v in slotmatched2.items():
        smatched2 += len(set(v))
    
    print ('ESHer upper bound')
    print (f'# of Event Type:\tHuman: {len(schema)}\tDisco.: {disco1}\tOverlap: {len(set(matched_type1))/len(schema)*100}\tAccpetable: {accept1/disco1*100}')
    print (f'# of Event Slot:\tHuman: {humanslot}\tDisco.: {slotnum1}\tOverlap: {smatched1/humanslot*100}\tAccpetable: {slotaccept1/slotnum1*100}\tRecall: {recall1/disco1*100}')
    
    print ('ESHer')
    print (f'# of Event Type:\tHuman: {len(schema)}\tDisco.: {len(data["final"])}\tOverlap: {len(set(matched_type2))/len(schema)*100}\tAccpetable: {accept2/len(data["final"])*100}')
    print (f'# of Event Slot:\tHuman: {humanslot}\tDisco.: {slotnum2}\tOverlap: {smatched2/humanslot*100}\tAccpetable: {slotaccept2/slotnum2*100}\tRecall: {recall2/len(data["final"])*100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ann', choices=['ann', 'evl'])
    parser.add_argument('--file', default='lm_output/-ann-covid19.json')
    parser.add_argument('--schema', default=None)
    args = parser.parse_args()
    print(vars(args))
    
    if args.mode == 'ann':
        annotate(args.file, args.schema)
    elif args.mode == 'evl':
        evaluate(args.file, args.schema)
    
    print('Done')