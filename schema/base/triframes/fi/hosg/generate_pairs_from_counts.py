#!/usr/bin/python3
import sys


def main():
    scale = True
    errors = 0
    for line in sys.stdin:
        res = line.strip().split()
        if len(res) != 4:
            errors += 1
            continue
        (frequency, verb, obj, subject) = res
        frequency = int(frequency)
        if scale:
            frequency = int(frequency / 10)
        for i in range(frequency):
            print(verb, obj, subject)
    print('Errors:', errors, file=sys.stderr)

if __name__ == '__main__':
    main()
