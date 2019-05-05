import numpy as np
import math
import matplotlib.pyplot as plt
import logistic_regressionex2 as lr
from sklearn.preprocessing import PolynomialFeatures

def main():
	data = np.loadtxt('ex2data2.txt', delimiter=',')
	#[1 for i in data]
#	x = [[1 for i in data],
#		[i[0] for i in data],
#		[i[1] for i in data],
#		[i[0]*i[1] for i in data],
#		[(i[0]**2)*(i[1]**2) for i in data],
#		[(i[0]**2)*i[1] for i in data],
#		[(i[0])*(i[1]**2) for i in data],
#	]
	x = [[1 for i in data],
		[i[0] for i in data],
		[i[1] for i in data],
		[i[0]**2 for i in data],
		[(i[0]**3) for i in data],
		[(i[0]**4) for i in data],
		[(i[0]**5) for i in data],
	]
	y = [i[2] for i in data]

	iterations = 5000
	alpha = 0.4
	lamb = 1

	theta, J = lr.logisticRegression(x,y,alpha,iterations,lamb)
	xo = list(range(200))
	xo = [(x/100)-1 for x in xo]
	x2 = [-(theta[0]+theta[1]*x1+theta[3]*x1**2+theta[4]*(x1**3)+theta[5]*x1**4+theta[6]*x1**5)/theta[2] for x1 in xo]


	fig, graf = plt.subplots()
	#fig2, j = plt.subplots()
	graf.scatter([x[0] for x in data if x[2]==0], [x[1] for x in data if x[2]==0], color='red', marker='x', label='descarte')
	graf.scatter([x[0] for x in data if x[2]==1], [x[1] for x in data if x[2]==1], color='blue', marker='o', label='aprovado')
	graf.plot(xo,x2,label='Hipótese',color='red', dashes=[6, 2])
	#j.plot(list(range(iterations)),J,label='Custo J x Iterações',color='green')
	
	graf.set_xlabel('Teste 1')
	graf.set_ylabel('Teste 2')
	fig.suptitle('Resultado de Testes em Microchip')
	graf.legend()
	plt.show()

if __name__ == '__main__':
	main()
