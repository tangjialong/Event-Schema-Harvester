#!/usr/bin/env groovy
import groovy.transform.Canonical
import groovy.transform.CompileStatic
import org.nlpub.watset.eval.NormalizedModifiedPurity
import org.nlpub.watset.util.Sampling

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

import static NormalizedModifiedPurity.normalize
import static NormalizedModifiedPurity.transform

Locale.setDefault(Locale.ROOT)

/*
 * Usage: groovy -classpath ../watset-java/target/watset.jar triframes_nmpu.groovy arguments.txt[.gz] fn-depcc-triples.tsv[.gz]
 */
def options = new CliBuilder().with {
    usage = 'triframes_nmpu.groovy arguments.txt[.gz] fn-depcc-triples.tsv[.gz]'

    t 'tabular format'
    p 'percentage format'
    s args: 1, 'sampling file'

    parse(args) ?: System.exit(1)
}

CLUSTER = Pattern.compile('^# Cluster *(.+?)$')
PREDICATES = Pattern.compile('^Predicates: *(.+)$')
SUBJECTS = Pattern.compile('^Subjects *(|\\(.+?\\)): *(.+)$')
OBJECTS = Pattern.compile('^Objects *(|\\(.+?\\)): *(.+)$')

@CompileStatic
@Canonical
class Element {
    String type
    String word
}

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

def arguments(path) {
    clusters = new HashMap<String, Set<Element>>()

    id = null

    lines(path).each { line ->
        if (line.empty) return

        matcher = CLUSTER.matcher(line)

        if (matcher.find()) {
            id = matcher.group(1)
            clusters[id] = new HashSet<Element>()
            return
        }

        matcher = PREDICATES.matcher(line)

        if (matcher.find()) {
            clusters[id].addAll(matcher.group(1).split(", ").collect { new Element('verb', it) })
            return
        }

        matcher = SUBJECTS.matcher(line)

        if (matcher.find()) {
            clusters[id].addAll(matcher.group(2).split(", ").collect { new Element('subject', it) })
            return
        }

        matcher = OBJECTS.matcher(line)

        if (matcher.find()) {
            clusters[id].addAll(matcher.group(2).split(", ").collect { new Element('object', it) })
            return
        }
    }

    return clusters.values()
}

FN_CLUSTER = Pattern.compile('^# *(.+?): .*$')

def framenet(path) {
    clusters = new HashMap<String, Set<Element>>()

    id = null

    lines(path).each { line ->
        if (line.empty) return

        matcher = FN_CLUSTER.matcher(line)

        if (matcher.find()) {
            id = matcher.group(1)
            if (!clusters.containsKey(id)) clusters.put(id, new HashSet<Element>())
            return
        }

        spo = line.split('\t', 3)

        clusters.get(id).add(new Element('subject', spo[0]))
        clusters.get(id).add(new Element('verb', spo[1]))
        clusters.get(id).add(new Element('object', spo[2]))
    }

    return clusters.values()
}

actual = arguments(Paths.get(options.arguments()[0]))
expected = framenet(Paths.get(options.arguments()[1]))

purity_pr = new NormalizedModifiedPurity<Element>()
purity_re = new NormalizedModifiedPurity<Element>(true, false)

format = options.p ? '%.2f\t%.2f\t%.2f' : '%.5f\t%.5f\t%.5f'

def extract(frames, type) {
    frames.collect { frame -> frame.grep { (it.type == type) } }
}

actual_verbs = normalize(transform(extract(actual, 'verb')))
expected_verbs = normalize(transform(extract(expected, 'verb')))

result = NormalizedModifiedPurity.evaluate(purity_pr, purity_re, actual_verbs, expected_verbs)
nmPU = result.precision * (options.p ? 100 : 1)
niPU = result.recall * (options.p ? 100 : 1)
f1 = result.f1Score * (options.p ? 100 : 1)

if (options.t) {
    printf(format + '\t', nmPU, niPU, f1)
} else {
    printf('Verb     nmPU/niPU/F1: ' + format + '%n', nmPU, niPU, f1)
}

actual_subjects = normalize(transform(extract(actual, 'subject')))
expected_subjects = normalize(transform(extract(expected, 'subject')))

result = NormalizedModifiedPurity.evaluate(purity_pr, purity_re, actual_subjects, expected_subjects)
nmPU = result.precision * (options.p ? 100 : 1)
niPU = result.recall * (options.p ? 100 : 1)
f1 = result.f1Score * (options.p ? 100 : 1)

if (options.t) {
    printf(format + '\t', nmPU, niPU, f1)
} else {
    printf('Subject  nmPU/niPU/F1: ' + format + '%n', nmPU, niPU, f1)
}

actual_objects = normalize(transform(extract(actual, 'object')))
expected_objects = normalize(transform(extract(expected, 'object')))

result = NormalizedModifiedPurity.evaluate(purity_pr, purity_re, actual_objects, expected_objects)
nmpu = result.precision * (options.p ? 100 : 1)
nipu = result.recall * (options.p ? 100 : 1)
f1 = result.f1Score * (options.p ? 100 : 1)

if (options.t) {
    printf(format + '\t', nmpu, nipu, f1)
} else {
    printf('Object   nmPU/niPU/F1: ' + format + '%n', nmpu, nipu, f1)
}

actual_frames = transform(actual)
expected_frames = normalize(transform(expected))

result = NormalizedModifiedPurity.evaluate(purity_pr, purity_re, normalize(actual_frames), expected_frames)
nmpu = result.precision * (options.p ? 100 : 1)
nipu = result.recall * (options.p ? 100 : 1)
f1 = result.f1Score * (options.p ? 100 : 1)

if (options.t) {
    printf(format + '%n', nmpu, nipu, f1)
} else {
    printf('Triframe nmPU/niPU/F1: ' + format + '%n', nmpu, nipu, f1)
}

if (options.s) {
    random = new Random(1337)

    dataset = actual_frames.toArray(new Map<String, Double>[0])
    f1_samples = new double[1000]

    System.err.print('Bootstrapping')

    for (i = 0; i < f1_samples.length; i++) {
        sample = normalize(Sampling.sample(dataset, random))
        result = NormalizedModifiedPurity.evaluate(purity_pr, purity_re, sample, expected_frames)
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
