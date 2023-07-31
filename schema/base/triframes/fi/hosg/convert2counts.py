#!/usr/bin/python3

import sys

def main():
    errors = 0
    for line in sys.stdin:
        res = line.strip().split()
        if len(res) != 4:
            errors += 1
            continue
        (verb, subject, obj, frequency) = res
        frequency = int(float(frequency))
        print(frequency, verb.split('#')[0], obj.split('#')[0], subject.split('#')[0])
    print('Errors:', errors, file=sys.stderr)


if __name__ == '__main__':
    main()
