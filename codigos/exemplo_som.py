from som import SOM
from entrada import Entrada
import random
import matplotlib.pyplot as plt
import numpy as np

# Gera 50 pontos ao redor de um grupo
entradas = []
for i in range(50):
    x = random.gauss(0, 0.1)
    y = random.gauss(0, 0.1)
    e = Entrada (valores_brutos=[x,y],limites=[[0,1],[0,1]])
    entradas.append(e)

# Gera 50 pontos ao redor de outro grupo 
for i in range(50):
    x = random.gauss(1, 0.1)
    y = random.gauss(1, 0.1)
    e = Entrada (valores_brutos=[x,y],limites=[[0,1],[0,1]])
    entradas.append(e)

som = SOM (entradas, linhas=2, colunas=2, taxa_aprendizado=0.1, numero_iteracoes=30, decaimento='não-linear')
print('Treinando...')
som.treinar()

labels_neuronios = []
x_vals_neuronios = []
y_vals_neuronios = []
x_vals_entradas = []
y_vals_entradas = []

for linha in som.network:
    for neuronio in linha:
        labels_neuronios.append (f'{neuronio.linha,neuronio.coluna}')
        x_vals_neuronios.append (neuronio.w[0])
        y_vals_neuronios.append (neuronio.w[1])

        for entrada in neuronio.entradas:
            x_vals_entradas.append (entrada.x[0])
            y_vals_entradas.append (entrada.x[1])

x_vals_neuronios = np.array(x_vals_neuronios)
y_vals_neuronios = np.array(y_vals_neuronios)
x_vals_entradas = np.array(x_vals_entradas)
y_vals_entradas = np.array(y_vals_entradas)


plt.scatter(x_vals_entradas, y_vals_entradas, color='red', label='Entradas')
plt.scatter(x_vals_neuronios, y_vals_neuronios)

for i, label in enumerate(labels_neuronios):
    plt.text(x_vals_neuronios[i], y_vals_neuronios[i], label, color='darkblue', ha='right')

plt.show()
