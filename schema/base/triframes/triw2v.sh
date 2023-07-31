#!/bin/bash -x

export LANG=en_US.UTF-8 LC_COLLATE=C
set -o pipefail -e

export CLASSPATH=$PWD/watset.jar
export WEIGHT=0

for setup in triples triples-prepless; do
  export VSO=depcc-common-$setup.tsv
  GOLD=fn-depcc-$setup.tsv

  for N in 5 10 30 50 100; do
    make triw2v.txt N=$N
    DATA=triw2v-n$N-$setup
    mv triw2v.txt $DATA.txt
    nice groovy 'fi/eval/triframes_nmpu.groovy' -t "$DATA.txt" "$GOLD" | tee "$DATA.nmpu"

    make triw2v-watset.txt N=$N
    DATA=triw2v-watset-n$N-top-top-$setup
    mv triw2v-watset.txt $DATA.txt
    nice groovy 'fi/eval/triframes_nmpu.groovy' -t "$DATA.txt" "$GOLD" | tee "$DATA.nmpu"

    make triw2v-watset.txt N=$N WATSET="-l mcl -g mcl-bin -gp bin=../mcl-14-137/bin/mcl"
    DATA=triw2v-watset-n$N-mcl-mcl-$setup
    mv triw2v-watset.txt $DATA.txt
    nice groovy 'fi/eval/triframes_nmpu.groovy' -t "$DATA.txt" "$GOLD" | tee "$DATA.nmpu"
  done
done
