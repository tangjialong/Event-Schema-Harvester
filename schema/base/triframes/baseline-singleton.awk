#!/usr/bin/awk -f

BEGIN {
  FS = OFS = "\t";
}

{
  sub(/#.+$/, "", $1);
  sub(/#.+$/, "", $2);
  sub(/#.+$/, "", $3);

  print "# Cluster " NR "\n";
  print "Predicates: " $1;
  print "Subjects: " $2;
  print "Objects: " $3 "\n";
}
