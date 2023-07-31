#!/bin/bash -ex

export LANG=en_US.UTF-8 LC_COLLATE=C
export JAVA_OPTS="$JAVA_OPTS -Xms64G -Xmx64G"

WATSET="$WORK/watset-java/target/watset.jar"

FRAMES_NMPU="fi/eval/triframes_nmpu.groovy"
VERBS_NMPU="fi/eval/verbs_nmpu.groovy"

FRAMES_GOLD="fn-depcc-triples.tsv"
VERBS_GOLD="$WORK/acl2014-dk-verb-classes/gold/korhonen2003.poly.txt"

echo -n > results-verbs.txt

for FRAMES in lda-frames.txt triw2v-watset-n30-top-top-triples.txt noac-1000-0_25.txt triw2v-watset-n30-mcl-mcl-triples.txt trispectral-k500-triples.txt hosg-kmeans-300-10-10000.txt trikmeans-k500-triples.txt triw2v-n30-triples.txt whole.txt singletons.txt ; do
    VERBS_SAMPLES="${FRAMES%.txt}-verbs.ser"

    echo "# $FRAMES" >> results-verbs.txt
    time nice groovy -classpath "$WATSET" "$VERBS_NMPU" -p -s "$VERBS_SAMPLES" "$FRAMES" "$VERBS_GOLD" | tee -a results-verbs.txt
    echo >> results-verbs.txt
done

echo -n > results.txt

for FRAMES in triw2v-watset-n30-top-top-triples.txt triw2v-watset-n30-mcl-mcl-triples.txt hosg-kmeans-300-10-10000.txt noac-1000-0_25.txt trispectral-k500-triples.txt trikmeans-k500-triples.txt lda-frames.txt triw2v-n30-triples.txt singletons.txt whole.txt ; do
    FRAMES_SAMPLES="${FRAMES%.txt}.ser"

    echo "# $FRAMES" >> results.txt
    time nice groovy -classpath "$WATSET" "$FRAMES_NMPU" -p -s "$FRAMES_SAMPLES" "$FRAMES" "$FRAMES_GOLD" | tee -a results.txt
    echo >> results.txt
done
