import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# declare gloabl variables
dataSet=[[5,2],[2,5], [1,5]]
targetArray = [1,-1, -1]
pMatrix = []
N = 3
classA = []
classB = []
C = 10000


# helper function for objective
def pMatrixGen():
	global pMatrix 
	pMatrix = [[0 for i in range(len(targetArray))] for j in range(len(targetArray))]
	for x in range(len(targetArray)):
		for y in range(len(targetArray)):
			pMatrix[x][y] = targetArray[x]*targetArray[y]*linearKernel(dataSet[x],dataSet[y])


def objective(alpha):
	innerMatrix = [[alpha[x] * alpha[y] for x in range(len(alpha))] for y in range(len(alpha))]
	innerMatrix =  np.multiply(innerMatrix, pMatrix)

	return 0.5*np.sum(innerMatrix) - np.sum(alpha);

# calculates the value which should be constrained to zero, returns scalar
def zerofun(alpha):
	return np.dot(alpha, targetArray)

# calculates the linear kernel
def linearKernel(point1, point2):
  	return np.dot(point1, point2)

#actually polynomial kernal with p = 2
def polyKernel(point1, point2):
     	return math.pow(np.dot(point1, point2)+1, 2)

#actually Radial Basis Function kernels, with sigma = 1
def radialKernel(point1, point2):
    	return math.exp(math.pow(np.linalg.norm(np.subtract(point1,point2)),2)*-1/8)

# return non-zero alpha in triples [(alphaValue, datapoint, targetValue)...]
def extractAlpha(alpha):
	nonZero = []
	for x in range(len(alpha)):
		if alpha[x] > math.pow(10, -5):
			nonZero.append((alpha[x], dataSet[x], targetArray[x]))
	return nonZero

# calculates the threshold b
def bValue(nonZero):
	if len(nonZero) == 0:
		print("nonZero array is empty")
		return 0
	bVector = []
	# We chose SV number 1
	for x in range(len(nonZero)):
		bVector.append(nonZero[x][0]*nonZero[x][2]*linearKernel(nonZero[0][1], nonZero[x][1]))
	return np.sum(bVector) - nonZero[0][2]

# indicator function, classify the datapoint 
def indicator(datapoint, nonZero, bValue):
	indVector = []
	for x in range(len(nonZero)):
		indVector.append(nonZero[x][0]*nonZero[x][2]*linearKernel(datapoint, nonZero[x][1]))

	return np.sum(indVector) - bValue

# Generate random dataset
def dataGenerator():

	global classA
	global classB
	global N
	global dataSet
	global targetArray

	classA = np.concatenate((np.random.randn(10, 2)*0.4 + [1.5, 0.5], np.random.randn(10, 2)*0.4 + [-1.5, 0.5]))
	classB = np.concatenate((np.random.randn(10, 2) * 0.4 + [0.0 , -0.5], np.random.randn(10, 2) * 0.4 + [4 , -2], np.random.randn(10, 2) * 0.4 + [-3 , -2] ))

	inputs = np.concatenate((classA , classB))
	targets = np.concatenate((np.ones(classA.shape[0]) , -np.ones(classB.shape[0])))

	
	N = inputs.shape[0] # Number of rows (samples)

	permute=list(range(N)) 
	random.shuffle(permute) 
	dataSet = inputs[permute, :]
	targetArray = targets[permute]



if __name__ == '__main__':
	np.random.seed(100)
	dataGenerator()
	pMatrixGen()
	start = np.zeros(N) 
	
	# run the minimize algorithm
	ret = minimize(objective, start, bounds=[(0, C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun})
	alpha = ret['x']
	nonZero = extractAlpha(alpha)
	b = bValue(nonZero)


	#plot the dataset and the support vectors
	plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b. ')
	plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r. ')

	xgrid = np.linspace(-5,5)
	ygrid = np.linspace(-4,4)

	grid = np.array([[indicator([x,y], nonZero, b) for x in xgrid] for y in ygrid])

	plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=("red", "black", "blue"), line=(1,3,1))

	plt.axis('equal')
	plt.show()
	




	
	




