#!/usr/bin/awk -f

BEGIN {
    OFS = "\t";
}

FNR == 1 {
    source = FILENAME;
    sub(/\.nmpu.*$/, ".txt", source);
    gsub(/\s+/, "\t");
    print FILENAME, $0, ENVIRON["HOSTNAME"] ":~" ENVIRON["USER"] "/frame-induction/" source;
}
