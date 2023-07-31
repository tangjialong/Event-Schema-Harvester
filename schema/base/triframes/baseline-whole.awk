#!/usr/bin/awk -f

BEGIN {
  FS = OFS = "\t";
}

{
  sub(/#.+$/, "", $1);
  sub(/#.+$/, "", $2);
  sub(/#.+$/, "", $3);

  predicates[$1];
  subjects[$2];
  objects[$3];
}

END {
  print "# Cluster 1\n";

  printf("Predicates: ");
  sep = "";
  for (word in predicates) {
    printf(sep word);
    sep = ", ";
  }
  print "";

  printf("Subjects: ");
  sep = "";
  for (word in subjects) {
    printf(sep word);
    sep = ", ";
  }
  print "";

  printf("Objects: ");
  sep = "";
  for (word in objects) {
    printf(sep word);
    sep = ", ";
  }
  print "\n";
}
