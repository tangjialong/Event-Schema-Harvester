#!/usr/bin/env groovy

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

/*
 * Copyright 2019 Dmitry Ustalov
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/*
 * Usage: groovy triframes_random100.groovy arguments.txt[.gz]
 */
def options = new CliBuilder().with {
    usage = 'triframes_random100.groovy arguments.txt[.gz]'

    s args: 1, 'sample size'
    r args: 1, 'random seed'

    parse(args) ?: System.exit(1)
}

STOPWORDS = ['all', 'another', 'any', 'anybody', 'anyone', 'anything', 'both', 'each', 'either',
            'enough', 'everybody', 'everyone', 'everything', 'few', 'he', 'her', 'hers', 'herself',
            'him', 'himself', 'his', 'i', 'in', 'it', 'its', 'itself', 'little', 'many', 'me', 'mine',
            'more', 'most', 'much', 'myself', 'neither', 'nobody', 'none', 'no one', 'nothing', 'one',
            'other', 'others', 'ours', 'ourselves', 'out', 'several', 'she', 'some', 'somebody', 'someone',
            'something', 'such', 'that', 'theirs', 'them', 'themselves', 'these', 'they', 'this',
            'those', 'us', 'we', 'what', 'whatever', 'which', 'whichever', 'who', 'whoever', 'whom',
            'whomever', 'whose', 'you', 'yours', 'yourself'] as Set

CLUSTER = Pattern.compile('^# Cluster *(.+?)$')
PREDICATES = Pattern.compile('^Predicates: *(.+)$')
SUBJECTS = Pattern.compile('^Subjects *(|\\(.+?\\)): *(.+)$')
OBJECTS = Pattern.compile('^Objects *(|\\(.+?\\)): *(.+)$')

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
    clusters = [:]

    id = null

    lines(path).each { line ->
        if (line.empty) return

        matcher = CLUSTER.matcher(line)

        if (matcher.find()) {
            id = matcher.group(1)
            clusters[id] = [verbs: new LinkedHashSet(), subjects: new LinkedHashSet<>(), objects: new LinkedHashSet<>()]
            return
        }

        matcher = PREDICATES.matcher(line)

        if (matcher.find()) {
            clusters[id]['verbs'].addAll(matcher.group(1).split(", ").collect { it.toLowerCase() })
            return
        }

        matcher = SUBJECTS.matcher(line)

        if (matcher.find()) {
            clusters[id]['subjects'].addAll(matcher.group(2).split(", ").collect { it.toLowerCase() })
            return
        }

        matcher = OBJECTS.matcher(line)

        if (matcher.find()) {
            clusters[id]['objects'].addAll(matcher.group(2).split(", ").collect { it.toLowerCase() })
            return
        }
    }

    return clusters
}

triframes = arguments(Paths.get(options.arguments()[0]))

System.err.println(triframes.size() + ' triframe(s) found in the dataset')

payload = triframes.grep {
    // Yes, we modify the map values here, but why not.
    it.value['subjects'].removeAll(STOPWORDS)
    it.value['objects'].removeAll(STOPWORDS)

    it.value['subjects'].size() > 1 && it.value['verbs'].size() > 1 && it.value['objects'].size() > 1
}

System.err.println(payload.size() + ' non-trivial triframe(s) found')

size = Math.min(options.s ? Integer.valueOf(options.s) : 100, payload.size)
System.err.println('Sample size is ' + size)

random = new Random(options.r ? Integer.valueOf(options.r) : 1337)

printf('id\tvote\tsubjects\tverbs\tobjects%n')

payload.with { Collections.shuffle(it, random); it }.take(size).each {
    printf('%s\t\t%s\t%s\t%s%n',
            it.key,
            it.value['subjects'].unique().join(', '),
            it.value['verbs'].unique().join(', '),
            it.value['objects'].unique().join(', '))
}
