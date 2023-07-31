#!/usr/bin/env groovy

Locale.setDefault(Locale.ROOT)

@Grab('org.apache.commons:commons-csv:1.6')
import org.apache.commons.csv.CSVParser
import static org.apache.commons.csv.CSVFormat.EXCEL

@Grab('org.dkpro.statistics:dkpro-statistics-agreement:2.1.0')
import org.dkpro.statistics.agreement.coding.*
import org.dkpro.statistics.agreement.distance.NominalDistanceFunction

import java.nio.file.Paths

items = [:]
subjects = [:]
verbs = [:]
objects = [:]

args.eachWithIndex { filename, rater ->
    Paths.get(filename).withReader { reader ->
        csv = new CSVParser(reader, EXCEL.withHeader().withDelimiter('\t' as char))

        for (record in csv.iterator()) {
            if (!(item   = record.get('id')))   continue
            if (!(answer = record.get('vote'))) continue
            if (!(subjectsList = record.get('subjects'))) continue
            if (!(verbsList = record.get('verbs'))) continue
            if (!(objectsList = record.get('objects'))) continue

            if (!items.containsKey(item)) items[item] = [:]

            items[item][rater] = answer.equalsIgnoreCase('1')
            subjects[item] = subjectsList
            verbs[item] = verbsList
            objects[item] = objectsList
        }
    }
}

def aggregate(answers) {
    answers.sort().inject([:]) { counts, _, answer ->
        counts[answer] = counts.getOrDefault(answer, 0) + 1
        counts
    }
}

def major(counts) {
    winner = counts.max { it.value }
    [winner.key, winner.value]
}

aggregated = items.inject([:]) { output, key, answers ->
    counts = aggregate(answers)
    (winner, count) = major(counts)
    output[key] = ['counts': counts, 'major': winner, 'unanimously': count == answers.size()]
    output
}

System.err.println('Unanimously good: ' + aggregated.values().grep { info ->
    info['major'] && info['unanimously']
}.size())

System.err.println('Total good: ' + aggregated.values().grep { info ->
    info['major']
}.size())

System.err.println('Unanimously bad: ' + aggregated.values().grep { info ->
    !info['major'] && info['unanimously']
}.size())

System.err.println('Total bad: ' + aggregated.values().grep { info ->
    !info['major']
}.size())

study = new CodingAnnotationStudy(args.size())

items.each { item, answers ->
    study.addItemAsArray(answers.inject(new Boolean[args.size()]) { array, rater, answer ->
        array[rater] = answer
        array
    })
}

percent = new PercentageAgreement(study)
System.err.printf('PercentageAgreement: %f %%%n', percent.calculateAgreement() * 100)

alphaNominal = new KrippendorffAlphaAgreement(study, new NominalDistanceFunction())
System.err.printf('KrippendorffAlphaAgreement: %f%n', alphaNominal.calculateAgreement())

fleissKappa = new FleissKappaAgreement(study)
System.err.printf('FleissKappaAgreement: %f%n', fleissKappa.calculateAgreement())

randolphKappa = new RandolphKappaAgreement(study)
System.err.printf('RandolphKappaAgreement: %f%n', randolphKappa.calculateAgreement())

printf('id\tmajor\tclass\tvotes1\tvotes0\tsubjects\tverbs\tobjects%n')

items.sort { aggregated[it.key]['unanimously'] ? (aggregated[it.key]['major'] ? 2 : 1) : 0 }.each {
    printf('%s\t%d\t%s\t%d\t%d\t%s\t%s\t%s%n',
            it.key,
            aggregated[it.key]['major'] ? 1 : 0,
            aggregated[it.key]['unanimously'] ? (aggregated[it.key]['major'] ? 'good' : 'bad') : 'clash',
            aggregated[it.key]['counts'].getOrDefault(true, 0),
            aggregated[it.key]['counts'].getOrDefault(false, 0),
            subjects[it.key],
            verbs[it.key],
            objects[it.key],
    )
}
