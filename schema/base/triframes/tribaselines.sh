#!/bin/bash -x

export LANG=en_US.UTF-8 LC_COLLATE=C
set -o pipefail -e

export CLASSPATH=$PWD/watset.jar
export WEIGHT=0

for setup in triples triples-prepless; do
  export VSO=depcc-common-$setup.tsv
  GOLD=fn-depcc-$setup.tsv

  for K in 10 150 500 1500 3000; do
    make K=$K trikmeans.txt
    DATA=trikmeans-k$K-$setup
    mv trikmeans.txt $DATA.txt
    nice groovy 'fi/eval/triframes_nmpu.groovy' -t "$DATA.txt" "$GOLD" | tee "$DATA.nmpu"
  done

  for K in 10 150 500; do
    make K=$K trispectral.txt
    DATA=trispectral-k$K-$setup
    mv trispectral.txt $DATA.txt
    nice groovy 'fi/eval/triframes_nmpu.groovy' -t "$DATA.txt" "$GOLD" | tee "$DATA.nmpu"
  done

  make tridbscan.txt
  DATA=tridbscan-$setup
  mv tridbscan.txt $DATA.txt
  nice groovy 'fi/eval/triframes_nmpu.groovy' -t "$DATA.txt" "$GOLD" | tee "$DATA.nmpu"
done
