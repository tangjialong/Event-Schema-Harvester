export LANG=en_US.UTF-8
export LC_COLLATE=C
export CLASSPATH=$(PWD)/watset.jar

VSO ?= vso-1.3m-pruned-strict.csv
WEIGHT ?= 10

ifdef W2V
W2VARG = --w2v=$(W2V)
else
ifdef PYRO
W2VARG = --pyro=$(PYRO)
else
W2VARG = --pyro=PYRO:w2v@localhost:9090
endif
endif

-include Makefile.local

triframes: triw2v.txt triw2v-watset.txt

N ?= 10

.PHONY: triw2v.txt
triw2v.txt:
	nice ./triw2v.py $(W2VARG) -n=$(N) --min-weight=$(WEIGHT) $(VSO) > $@

.PHONY: trihosg.txt
trihosg.txt:
	nice ./trihosg.py -n=$(N) fi/hosg/300-10/{word,context,relation}-matrix fi/hosg/counts.gz > $@

WATSET ?= -l cw -lp 'mode=top' -g cw -gp 'mode=top'

.PHONY: triw2v-watset.txt
triw2v-watset.txt:
	nice ./triw2v.py $(W2VARG) -n=$(N) --min-weight=$(WEIGHT) --pickle $(@:txt=pkl) $(VSO)
	nice groovy ./triw2v_watset.groovy $(WATSET) $(@:txt=pkl) > $@

.PHONY: trihosg-watset.txt
trihosg-watset.txt:
	nice ./trihosg.py $(W2VARG) -n=$(N) --pickle $(@:txt=pkl) fi/hosg/300-10/{word,context,relation}-matrix fi/hosg/counts.gz
	nice groovy ./triw2v_watset.groovy $(WATSET) $(@:txt=pkl) > $@

K ?= 10

.PHONY: trikmeans.txt
trikmeans.txt:
	nice ./triclustering.py $(W2VARG) --method=kmeans -k=$(K) --min-weight=$(WEIGHT) $(VSO) > $@

.PHONY: trispectral.txt
trispectral.txt:
	nice ./triclustering.py $(W2VARG) --method=spectral -k=$(K) --min-weight=$(WEIGHT) $(VSO) > $@

.PHONY: tridbscan.txt
tridbscan.txt:
	nice ./triclustering.py $(W2VARG) --method=dbscan --min-weight=$(WEIGHT) $(VSO) > $@

depcc-common-triples-full-singleton.tsv: depcc-common-triples-full.tsv
	gawk -f ./baseline-singleton.awk $< > $@

depcc-common-triples-full-whole.tsv: depcc-common-triples-full.tsv
	gawk -f ./baseline-whole.awk $< > $@

lda-frames.txt:
	./lda-frames.py ../verb-clusters/lda-frames/data/bnc-export/{frames,roles,verbs}.txt > $@

clean:
	rm -rf __pycache__
	rm -fv *.txt *-watset.tsv
	rm -fv *.nmpu *.nmpu_cross

watset.jar:
	curl -Lo "$@" https://github.com/nlpub/watset-java/releases/download/2.0.0-rc1/watset.jar

data: vso-1.3m-pruned-strict.csv depcc-common-triples-full.tsv fn-depcc-triples-full.tsv

vso-1.3m-pruned-strict.csv: vso-1.3m-pruned-strict.csv.gz
	gunzip -kf $< > $@

vso-1.3m-pruned-strict.csv.gz:
	curl -Lo "$@" http://panchenko.me/data/joint/verbs/cc16malt/vso-1.3m-pruned-strict.csv.gz

depcc-common-triples-full.tsv:
	gunzip -c fi/eval/data/$@.gz > $@

fn-depcc-triples-full.tsv:
	gunzip -c fi/eval/data/$@.gz > $@

GoogleNews-vectors-negative300.bin: GoogleNews-vectors-negative300.bin.gz
	gunzip -kf $< > $@

GoogleNews-vectors-negative300.bin.gz:
	curl -Lo "$@" https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
