from som import SOM
from entrada import Entrada
import random
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s',level=logging.INFO)

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
som.plotar(entradas)
logging.info('Treinando...')
som.treinar()
som.plotar(entradas)
