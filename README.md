# Harvesting Event Schemas from Large Language Models

### Catalogue
```bash
tree ./schema
├── base
│   ├── groovy-2.4.21
│   ├── triframes
│   ├── __init__.py
│   ├── baseline_agglo.py
│   ├── baseline_etyeclus.py
│   ├── baseline_jcsc.py
│   ├── baseline_kmeans.py
│   ├── baseline_triframes.py
│   └── evaluater.py
├── data
│   ├── ace05
│   │   └── train.oneie.json
│   ├── chfinann
│   │   └── train.json
│   ├── covid19
│   │   └── corpus.txt
│   ├── duee
│   │   └── train.json
│   ├── ere 
│   │   └── train.oneie.json
│   ├── nyt
│   │   └── LDC2008T19
│   ├── pandemic
│   │   └── corpus.txt
│   └── rmrb
│       └── rmrb-1946-2001.json
├── lm_output
│   ├── chfinann
│   ├── covid19
│   ├── ere
│   ├── nyt-5k
│   ├── pandemic
│   └── rmrb-5k
├── resources
│   ├── bert-large-uncased-whole-word-masking
│   ├── distiluse-base-multilingual-cased-v2
│   ├── en_verb_sense_dict_w_features.json
│   ├── enwiki-20220301-pages-articles-multistream.xml
│   └── zhwiki-20220301-pages-articles-multistream.xml
├── scripts
│   ├── annotator.py
│   ├── create_corpus.py
│   ├── create_resources.py
│   ├── extract.py
│   ├── preprocessor.py
│   └── wikiextractor.py
├── event_mention_cluster.py
├── harvester.py
├── prepare_in_context.py
├── README.md
├── requirement.txt
└── sentence_simplifier.py
```

### Prepare environment
```bash
conda create -n schema python=3.8
conda activate schema
pip install -r requirement.txt
```

### Prepare data & resources
d1. ace05: Following http://blender.cs.illinois.edu/software/oneie/
d2. chfinann: Download from https://github.com/dolphin-zs/Doc2EDAG
d3. covid19: Download from https://github.com/jmshen1994/ETypeClus
d4. duee: Download from https://ai.baidu.com/broad/subordinate?dataset=duee
d5. ere: Following http://blender.cs.illinois.edu/software/oneie/
d6. nyt: Download from https://catalog.ldc.upenn.edu/LDC2008T19
d7. pandemic:  Download from https://github.com/jmshen1994/ETypeClus
d8. rmrb: People's daily 1946-2003

o1. bert-large-uncased-whole-word-masking: Download from https://huggingface.co/bert-large-uncased-whole-word-masking
o2. distiluse-base-multilingual-cased-v2: Download from https://huggingface.co/distiluse-base-multilingual-cased-v2
o3. enwiki-20220301-pages-articles-multistream.xml: Download from https://dumps.wikimedia.org/enwiki/20220301/
o4. en_verb_sense_dict_w_features.json: Download from https://github.com/jmshen1994/ETypeClus
o5. zhwiki-20220301-pages-articles-multistream.xml: Download from https://dumps.wikimedia.org/zhwiki/20220301/

```bash
# Reform different data formats to json ---> [{sentence}]
# Results: data/ace05/corpus.json, data/chfinann/corpus.json, data/covid19/corpus.json, data/ere/corpus.json, data/nyt/corpus.json, data/nyt/corpus.5k.json, data/pandemic/corpus.json, data/rmrb/corpus.json, data/rmrb/corpus.5k.json
python scripts/create_corpus.py

# Reform XML wiki dump file to json ---> [{id, revid, url, title, text}]
# Results: resources/enwiki-20220301-json, resources/zhwiki-20220301-json
python scripts/wikiextractor.py -i resources/enwiki-20220301-pages-articles-multistream.xml -o resources/enwiki-20220301-json -j
python scripts/wikiextractor.py -i resources/zhwiki-20220301-pages-articles-multistream.xml -o resources/zhwiki-20220301-json -j

# Preprocess json file ---> [{text}]
# Results: resources/enwiki-20220301-json-preprocessed, resources/zhwiki-20220301-json-preprocessed
# Results: resources/enwiki.json, resources/zhwiki.json
# Results (for debugging): resources/enwiki-5Ksample.json, resources/zhwiki-5Ksample.json
python scripts/preprocessor.py -i resources/enwiki-20220301-json -o resources/enwiki-20220301-json-preprocessed -j resources -d enwiki
python scripts/preprocessor.py -i resources/zhwiki-20220301-json -o resources/zhwiki-20220301-json-preprocessed -j resources -d zhwiki

# Statistic verb/all lemma frequence
python scripts/create_resources.py --enwiki resources/enwiki.json --zhwiki resources/zhwiki.json
```

### Simplify Sentences
```bash
python sentence_simplifier.py --input_file data/chfinann/corpus.json --spacy_model zh_core_web_lg --verb_freq_file resources/zh_verb_lemma_freq.json --all_lemma_freq_file resources/zh_all_lemma_freq.json
python sentence_simplifier.py --input_file data/covid19/corpus.json --spacy_model en_core_web_lg --verb_freq_file resources/en_verb_lemma_freq.json --all_lemma_freq_file resources/en_all_lemma_freq.json
python sentence_simplifier.py --input_file data/ere/corpus.json --spacy_model en_core_web_lg --verb_freq_file resources/en_verb_lemma_freq.json --all_lemma_freq_file resources/en_all_lemma_freq.json
python sentence_simplifier.py --input_file data/nyt/corpus.5k.json --spacy_model en_core_web_lg --verb_freq_file resources/en_verb_lemma_freq.json --all_lemma_freq_file resources/en_all_lemma_freq.json
python sentence_simplifier.py --input_file data/pandemic/corpus.json --spacy_model en_core_web_lg --verb_freq_file resources/en_verb_lemma_freq.json --all_lemma_freq_file resources/en_all_lemma_freq.json
python sentence_simplifier.py --input_file data/rmrb/corpus.5k.json --spacy_model zh_core_web_lg --verb_freq_file resources/zh_verb_lemma_freq.json --all_lemma_freq_file resources/zh_all_lemma_freq.json
```

### Event Schema Discovery 
```bash
python prepare_incontext.py --incontext_corpus data/duee --data data/chfinann --mode 0
python prepare_incontext.py --incontext_corpus data/ace05 --data data/ere --mode 0

python harvester.py --input_file lm_output/chfinann --output_file lm_output/results-chfinann.json --annotate_file lm_output/ann-chfinann.json --data_file data/chfinann --language zh --link_sentence
python harvester.py --input_file lm_output/ere --output_file lm_output/results-ere.json --data_file data/ere --language en

python scripts/annotator.py --mode ann --file lm_output/ann-chfinann.json --schema data/chfinann/schema.json
python scripts/annotator.py --mode ann --file lm_output/ann-ere.json --schema data/ere/schema.json

python scripts/annotator.py --mode evl --file lm_output/ann-chfinann.json --schema data/chfinann/schema.json
python scripts/annotator.py --mode evl --file lm_output/ann-ere.json --schema data/ere/schema.json
```

### Event Mention Clustering
```bash
python prepare_incontext.py --incontext_corpus data/ace05 --data data/ere --mode 0 --top_num 0
export GROOVY_HOME=/shared_home/tangjialong/Projects/schema/base/groovy-2.4.21
export PATH=$GROOVY_HOME/bin:$PATH
python event_mention_cluster.py --ere_dir data/ere --dict_file resources/en_verb_sense_dict_w_features.json --gpu_id 0 --num_sample 3
```

### Others
```bash
python prepare_incontext.py --incontext_corpus data/duee --data data/chfinann --mode 0
python prepare_incontext.py --incontext_corpus data/ace05 --data data/covid19 --mode 0
python prepare_incontext.py --incontext_corpus data/ace05 --data data/ere --mode 0
python prepare_incontext.py --incontext_corpus data/ace05 --data data/nyt --mode 0
python prepare_incontext.py --incontext_corpus data/ace05 --data data/pandemic --mode 0
python prepare_incontext.py --incontext_corpus data/duee --data data/rmrb --mode 0   # 3025

python harvester.py --input_file lm_output/chfinann --output_file lm_output/results-chfinann.json --annotate_file lm_output/ann-chfinann.json --data_file data/chfinann --language zh --link_sentence
python harvester.py --input_file lm_output/covid19 --output_file lm_output/results-covid19.json --annotate_file lm_output/ann-covid19.json --data_file data/covid19 --language en --link_sentence
python harvester.py --input_file lm_output/ere --output_file lm_output/results-ere.json --annotate_file lm_output/ann-ere.json --data_file data/ere --language en --link_sentence
python harvester.py --input_file lm_output/nyt-5k --output_file lm_output/results-nyt.json --annotate_file lm_output/ann-nyt.json --data_file data/nyt --language en --link_sentence
python harvester.py --input_file lm_output/pandemic --output_file lm_output/results-pandemic.json --annotate_file lm_output/ann-pandemic.json --data_file data/pandemic --language en --link_sentence
python harvester.py --input_file lm_output/rmrb-5k --output_file lm_output/results-rmrb.json --annotate_file lm_output/ann-rmrb.json --data_file data/rmrb --language zh --link_sentence

python scripts/annotator.py --file lm_output/ann-chfinann.json --schema data/chfinann/schema.json --mode evl
python scripts/annotator.py --file lm_output/ann-ere.json --schema data/ere/schema.json --mode evl
```