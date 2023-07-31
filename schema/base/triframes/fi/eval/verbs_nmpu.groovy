#!/usr/bin/env groovy
import org.nlpub.watset.eval.NormalizedModifiedPurity
import org.nlpub.watset.util.Sampling

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

import static NormalizedModifiedPurity.transform
import static NormalizedModifiedPurity.normalize

Locale.setDefault(Locale.ROOT)

/*
 * Usage: groovy -classpath ../watset-java/target/watset.jar verbs_nmpu.groovy arguments.txt[.gz] korhonen2003.poly.txt[.gz]
 */
def options = new CliBuilder().with {
    usage = 'verbs_nmpu.groovy arguments.txt[.gz] korhonen2003.poly.txt[.gz]'

    t 'tabular format'
    p 'percentage format'
    s args: 1, 'sampling file'

    parse(args) ?: System.exit(1)
}

CLUSTER = Pattern.compile('^# Cluster *(.+?)$')
PREDICATES = Pattern.compile('^Predicates: *(.+)$')

def lines(path) {
    if (!path.toString().endsWith(".gz")) return Files.lines(path)

    Files.newInputStream(path).with { is ->
        new GZIPInputStream(is).with { gis ->
            new InputStreamReader(gis, StandardCharsets.UTF_8).with { reader ->
                new BufferedReader(reader).with { br ->
                    return br.lines()
                }
            }
        }
    }
}

TAB = Pattern.compile('\t+')

def korhonen(path) {
    clusters = new ArrayList<Set<String>>()

    lines(path).each { line ->
        if (line.empty) return

        row = TAB.split(line, 3)

        words = new HashSet<>(Arrays.asList(row[2].split(' ')))

        clusters.add(words)
    }

    return clusters
}

def arguments(path, expected) {
    lexicon = expected.flatten().toSet()

    clusters = new HashMap<String, Set<String>>()

    id = null

    lines(path).each { line ->
        if (line.empty) return

        matcher = CLUSTER.matcher(line)

        if (matcher.find()) {
            id = matcher.group(1)
            return
        }

        matcher = PREDICATES.matcher(line)

        if (matcher.find()) {
            words = matcher.group(1).split(", ").grep(lexicon).toSet()
            if (!words.isEmpty()) clusters[id] = words
            return
        }
    }

    return clusters.values()
}

expected_raw = korhonen(Paths.get(options.arguments()[1]))
actual_raw = arguments(Paths.get(options.arguments()[0]), expected_raw)

expected = normalize(transform(expected_raw))
actual = transform(actual_raw)

purity_pr = new NormalizedModifiedPurity<>()
purity_re = new NormalizedModifiedPurity<>(true, false)
result = NormalizedModifiedPurity.evaluate(purity_pr, purity_re, normalize(actual), expected)

format = options.p ? '%.2f\t%.2f\t%.2f%n' : '%.5f\t%.5f\t%.5f%n'

nmPU = result.precision * (options.p ? 100 : 1)
niPU = result.recall * (options.p ? 100 : 1)
f1 = result.f1Score * (options.p ? 100 : 1)

if (options.t) {
    printf(format, nmPU, niPU, f1)
} else {
    printf('nmPU/niPU/F1: ' + format, nmPU, niPU, f1)
}

if (options.s) {
    random = new Random(1337)

    dataset = actual.toArray(new Map<String, Double>[0])
    f1_samples = new double[10000]

    System.err.print('Bootstrapping')

    for (i = 0; i < f1_samples.length; i++) {
        sample = normalize(Sampling.sample(dataset, random))
        result = NormalizedModifiedPurity.evaluate(purity_pr, purity_re, sample, expected)
        f1_samples[i] = result.f1Score
        System.err.printf(' %d', i + 1)
        System.err.flush()
    }

    System.err.println()

    Files.newOutputStream(Paths.get(options.s)).withCloseable { fos ->
        new ObjectOutputStream(fos).withCloseable { oos ->
            oos.writeObject(f1_samples)
        }
    }
}
