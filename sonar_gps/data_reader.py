#!/usr/bin/env python
import numpy as np
import re

def read(filename='./unfreezed_rawdistances.log'):
    d = {}
    with open(filename) as f:
        line = f.readline()
        regex = re.compile('Beacon from (\d+) to (\d+) is (\d+)\[mm\]')

        while line:
            m = regex.match(line)
            line = f.readline()
            if m is None:
                continue

            t    = int(m.group(1)) # transmitter
            r    = int(m.group(2)) # receiver
            dist = int(m.group(3))
            if not t in d:
                d[t] = {}
            if not r in d[t]:
                d[t][r] = []
            d[t][r].append(dist/1000.0)


    return d
