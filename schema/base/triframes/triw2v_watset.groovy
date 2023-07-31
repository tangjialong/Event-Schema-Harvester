#!/usr/bin/env groovy
import groovy.transform.Canonical
import groovy.transform.CompileStatic

@Grab('net.razorvine:pyrolite:4.20')
import net.razorvine.pickle.Unpickler

import org.jgrapht.graph.DefaultWeightedEdge
import org.jgrapht.graph.SimpleWeightedGraph
import org.nlpub.watset.util.AlgorithmProvider
import org.nlpub.watset.util.CosineContextSimilarity
import org.nlpub.watset.graph.Watset

@CompileStatic
@Canonical
class Triple {
    String subject
    String predicate
    String object
}

def options = new CliBuilder().with {
    l 'local', longOpt: 'local', required: true, args: 1
    lp 'local-params', longOpt: 'local-params', required: false, args: 1
    g 'global', longOpt: 'global', required: true, args: 1
    gp 'global-params', longOpt: 'global-params', required: false, args: 1
    parse(args) ?: System.exit(1)
}

local = new AlgorithmProvider<Triple, DefaultWeightedEdge>(options.l as String, !options.lp ? Collections.emptyMap() : options.lp as String)
global = new AlgorithmProvider<Triple, DefaultWeightedEdge>(options.g as String, !options.gp ? Collections.emptyMap() : options.gp as String)

stream = new FileInputStream(options.arguments().get(0))
unpickler = new Unpickler()
edges = (List) unpickler.load(stream)

builder = SimpleWeightedGraph.<Triple, DefaultWeightedEdge> createBuilder(DefaultWeightedEdge.class)

edges.each {
    (sourceTuple, targetTuple, data) = it

    source = new Triple(sourceTuple[0] as String, sourceTuple[1] as String, sourceTuple[2] as String)
    target = new Triple(targetTuple[0] as String, targetTuple[1] as String, targetTuple[2] as String)
    weight = (data as HashMap).getOrDefault('weight', 0d) as Double

    builder.addVertices(source, target)

    if (source != target) builder.addEdge(source, target, weight.doubleValue())
}

graph = builder.build()

watset = new Watset<Triple, DefaultWeightedEdge>(graph, local, global, new CosineContextSimilarity<>())
watset.fit()

id = 0

watset.clusters.sort { -it.size() }.each { cluster ->
    printf('# Cluster %d\n\n', ++id)

    predicates = cluster.collect { it.predicate }.toSet().join(', ')
    subjects = cluster.collect { it.subject }.toSet().join(', ')
    objects = cluster.collect { it.object }.toSet().join(', ')

    printf('Predicates: %s\n', predicates)
    printf('Subjects: %s\n', subjects)
    printf('Objects: %s\n\n', objects)
}
