#!/usr/bin/env python3

import monkdata as m
import dtree as d
import drawtree_qt5 as dr
import random
import matplotlib.pyplot as plt
import numpy as np 


# Method to partition the dataset into training and validation
def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata)*fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

# calculate the best performance of the trees
def pruneTree(tree, validation):
	pruned = d.allPruned(tree)
	best = 0
	bestT = tree

	for x in pruned:
		value = d.check(x, validation)
		if(value > best):
			best = value
			bestT = x

	return best, bestT



if __name__ == '__main__':
	#initiate different fractions of training and validation data
	fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	meanArray = []
	stdArray = []
	resultArray = []
	bestScore = 0


	# for each fraction value Prune until we it does not improve 
	for x in fraction:
		for u in range(0,1000):
			newScore = 0
			trainData, valData = partition(m.monk2, x)
			oldTree = d.buildTree(trainData, m.attributes)
			bestScore = d.check(oldTree, valData)
			improved = True

			while improved:
				newScore, newTree = pruneTree(oldTree, valData)
				if(newScore > bestScore):
					bestScore = newScore
					oldTree = newTree
				else:
					improved = False


			resultArray.append(1-d.check(oldTree, m.monk2test))
		
		#Save the mean and variance
		meanArray.append(np.mean(resultArray))
		stdArray.append(np.var(resultArray))
		resultArray = []

		

	#Plot the graph with mean and variance
	fig, meanAxis = plt.subplots()

	#Describe the mean graph
	color = "tab:red"
	meanAxis.set_xlabel("fraction")
	meanAxis.set_ylabel("mean error", color=color)
	meanAxis.plot(fraction, meanArray, color=color)
	meanAxis.tick_params(axis="y", labelcolor=color)

	stdAxis = meanAxis.twinx()


	#Describe the standard deviation
	color="tab:blue"
	stdAxis.set_ylabel("Std", color=color)
	stdAxis.plot(fraction, stdArray, color=color)
	stdAxis.tick_params(axis="y", labelcolor=color)

	fig.tight_layout()
	plt.show()
    
    #dr.drawTree(bestT)



