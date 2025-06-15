from neuronio import Neuronio
import math
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s',level=logging.INFO)

class SOM:
    
    def __init__(self, entradas, linhas, colunas, numero_iteracoes, taxa_aprendizado, decaimento='linear'):
        self.numero_iteracoes = numero_iteracoes
        self.entradas = entradas
        self.taxa_aprendizado = taxa_aprendizado
        self.network = []
        self.linhas = linhas
        self.colunas = colunas
        self.decaimento = decaimento
        self.sigma_vizinhanca = 1
        for i in range(linhas):
            self.network.append([])
            for j in range (colunas):
                self.network [i].append (Neuronio (numero_atributos=len(entradas[0].x),linha=i, coluna=j, pesos_aleatorios=True))

    #Treina o SOM tentando aproximar o mapa às entradas de treinamento.
    def treinar(self):
        eta = self.taxa_aprendizado_constante()
        for k in range(self.numero_iteracoes):
            logging.info(f'\nIteração {k+1}:')
            for linha in self.network:
                celulas = ''
                for neuronio in linha:
                    celulas += (f'[{len(neuronio.entradas)}] \t')  
                    neuronio.entradas = []
                logging.info(celulas+ '\n')
            
            sigma = self.sigma_vizinhanca_linear(k)
            for i in range(len (self.entradas)):
                bmu = self.eleger_bmu(self.entradas[i])
                logging.debug(f'-> BMU: [{bmu.linha,bmu.coluna}]')
                self.atualizar_pesos(bmu, eta, sigma, self.entradas[i])

            if self.decaimento == 'linear':
                eta = self.taxa_aprendizado_linear(k)
            else:
                eta = self.taxa_aprendizado_nao_linear(k)
            
            logging.info(f'eta: {eta}, sigma:{sigma}')

    def taxa_aprendizado_linear(self, k):
        eta = self.taxa_aprendizado * (1 - (k/self.numero_iteracoes))
        return eta

    def taxa_aprendizado_nao_linear(self, k):
        eta = self.taxa_aprendizado * math.exp(- (k/(self.numero_iteracoes/2)))
        return eta

    def taxa_aprendizado_constante(self):
        return self.taxa_aprendizado
    
    def sigma_vizinhanca_nao_linear(self, k):
        sigma_0 = max(self.linhas, self.colunas) / 2
        tau = self.numero_iteracoes / 2
        return sigma_0 * math.exp(-k / tau)

    def sigma_vizinhanca_linear(self, k):
        sigma_0 = (1 - (k/self.numero_iteracoes))
        return sigma_0
    
    def sigma_vizinhanca_constante(self, k):
        return 1

    #Atualiza os pesos de todos os neurônios dependendo da proximidade ao BMU.
    def atualizar_pesos(self,bmu, eta, sigma, entrada):
        for linha in self.network:
            for neuronio in linha:
                r = math.sqrt(math.pow((bmu.linha - neuronio.linha),2) + math.pow((bmu.coluna - neuronio.coluna),2))
                taxa_vizinhanca = math.exp(- (r**2)/(2*sigma**2))
                logging.debug(f'-> theta: {taxa_vizinhanca} para neuronio {neuronio.linha, neuronio.coluna}')
                for i in range (len(entrada.x)):
                    neuronio.w[i] += taxa_vizinhanca * (entrada.x[i] - neuronio.w[i]) * eta
            
    def eleger_bmu(self, entrada):
        bmu = None
        d_bmu = -1
        for l in range(self.linhas):
            for c in range (self.colunas):
                soma = 0
                for i in range(len(entrada.x)):
                    soma += math.pow(entrada.x[i] - self.network[l][c].w[i],2)
                d = math.sqrt(soma) 
                if bmu is None or d < d_bmu:
                    d_bmu = d
                    bmu = self.network[l][c]
        bmu.entradas.append(entrada)
        return bmu 
    
    def plotar(self, entradas):
        labels_neuronios = []
        x_vals_neuronios = []
        y_vals_neuronios = []
        x_vals_entradas = []
        y_vals_entradas = []
        
        for linha in self.network:
            for neuronio in linha:
                labels_neuronios.append (f'{neuronio.linha,neuronio.coluna} - [{len(neuronio.entradas)}]')
                x_vals_neuronios.append (neuronio.w[0])
                y_vals_neuronios.append (neuronio.w[1])

        for entrada in entradas:
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