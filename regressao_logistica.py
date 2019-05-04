import numpy as np
import math
import matplotlib.pyplot as plt
import logistic_regression as lr

def main():
	data = np.loadtxt('ex2data1.txt', delimiter=',')
	x = [[1 for i in data], [i[0]/100 for i in data],[i[1]/100 for i in data]]
	y = [i[2] for i in data]

	iterations = 5000
	alpha = 0.4

	theta, J = lr.logisticRegression(x,y,alpha,iterations)
	print(theta)
	print(min(J))

	xo = list(range(60))
	xo = [(x/100)+0.3 for x in xo]
	Hx = [-(theta[0]+theta[1]*x1)/theta[2] for x1 in xo]


	fig, graf = plt.subplots()
	fig2, j = plt.subplots()
	graf.scatter([x[0]/100 for x in data if x[2]==0], [x[1]/100 for x in data if x[2]==0], color='red', label='reprovados')
	graf.scatter([x[0]/100 for x in data if x[2]==1], [x[1]/100 for x in data if x[2]==1], color='green', label='admitidos')
	graf.plot(xo,Hx,label='Hipótese',color='red', dashes=[6, 2])
	j.plot(list(range(iterations)),J,label='Custo J x Iterações',color='green')
	fig2.suptitle('Custo J x Iterações')
	graf.set_xlabel('Resultado do Exame 1')
	graf.set_ylabel('Resultado do Exame 2')
	graf.legend()
	plt.show()

if __name__ == '__main__':
	main()
