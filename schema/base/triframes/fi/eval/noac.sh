dataset="no-preps"
for f in `ls /home/panchenko/tmp/triclustering-results/${dataset}/*arguments.tsv` ; do
    echo $f
    groovy -classpath watset.jar triframes_nmpu.groovy -t $f data/fn-depcc-triples-${dataset}.tsv.gz
done
