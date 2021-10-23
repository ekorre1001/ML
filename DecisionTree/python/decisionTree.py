#!/usr/bin/env python3

import monkdata as m
import dtree as t
import drawtree_qt5 as d



if __name__ == '__main__':
    
	# Build the tree and check the performance
    x = t.buildTree(m.monk1, m.attributes)
    print(1-t.check(x, m.monk1test))
    # Draw the entire tree
    d.drawTree(x)
	
