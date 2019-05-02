import numpy as np
import matplotlib.pyplot as plt
import logistic_regression as lr

def main():
	data = np.loadtxt('ex2data1.txt', delimiter=',')
	x = [[i[0] for i in data],[i[1] for i in data],[i[2] for i in data]]
	fig, graf = plt.subplots()
	graf.scatter([x[0] for x in data if x[2]==0], [x[1] for x in data if x[2]==0], color='red', label='reprovados')
	graf.scatter([x[0] for x in data if x[2]==1], [x[1] for x in data if x[2]==1], color='green', label='admitidos')
	graf.set_xlabel('Resultado do Exame 1')
	graf.set_ylabel('Resultado do Exame 2')
	graf.legend()
	plt.show()

if __name__ == '__main__':
	main()
