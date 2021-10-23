#!/usr/bin/env python3

import monkdata as m
import dtree as d


if __name__ == '__main__':
	# Calculating the entropy
    #print("information gain är ", d.entropy(m.monk3))
    # Caculating the averageGain for all the attributes
    for x in range(0,6):
        print("information gain är på a", x, " är ", d.averageGain(m.monk3, m.attributes[x]))


