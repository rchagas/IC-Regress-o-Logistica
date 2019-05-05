import numpy as np
import math

def computeCost (X,Y,theta,lamb):
	m = len(Y)
	J = 0
	pred = [0 for i in range(m)]
	for i in range(m):
		for j in range(len(theta)):
			pred[i] += theta[j]*X[j][i]
	pred = [(1 / (1 + math.exp(-p))) for p in pred]	
	J = (-1/m) * sum([(y * math.log10(hx)) + ((1-y) * math.log10(1-hx)) for hx,y in zip(pred,Y)]) + (lamb/2*m)*sum([theta**2 for theta in theta])
	return J

def logisticRegression(X,Y,alpha,iteracoes,lamb):
	theta = [0 for i in range(len(X))]
	m = len(Y)
	jPrev = [0 for i in range(iteracoes)]
	thetaNew = [0 for i in range(len(theta))]
	for i in range(iteracoes):
		h = [0 for i in range(m)]
		for j in range(m):
			for z in range(len(theta)):
				h[j] += theta[z]*X[z][j]
		h = [1 / (1 + math.exp(-hx)) for hx in h]
		erro = [h-y for h,y in zip(h,Y)]
		for j in range(len(theta)):
			thetaNew[j] = theta[j] - alpha*(1/m) * sum([erro*x for x,erro in zip(X[j],erro)])
		theta = thetaNew
		
		jPrev[i] = computeCost(X,Y,theta,lamb)
	return theta, jPrev
