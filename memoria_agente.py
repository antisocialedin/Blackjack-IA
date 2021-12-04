import os
import sys
import pdb
import gym
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from gym.envs.toy_text import blackjack


# Classe para memória do agente
class Memoria():

    # Método Construtor
    def __init__(self, capacidade):

        # Inicializamos os atributos da classe
        # A capacidade será um parâmetro que definiremos durante o treinamento
        self.capacidade = capacidade
        self.memoria = []
        self.posicao = 0

    # Método para adicionar estado, ação e ação subsequente à memória. Usaremos tensores PyTorch para isso.
    def carrega_memoria(self, estado_corrente, acao, proximo_estado, recompensa, done):

        # Define os parâmetros como tensores PyTorch
        estado_corrente_t = torch.Tensor(estado_corrente)
        acao_t = torch.Tensor([acao])
        proximo_estado_t = torch.Tensor(proximo_estado)
        recompensa_t = torch.Tensor([recompensa])
        done_t = torch.Tensor([done])

        # Lista para transição de estado
        transicao_estado = [estado_corrente_t, acao_t,
                            recompensa_t, proximo_estado_t, done_t]

        # Verificamos se a memória ainda tem capacidade (capacidade é um parâmetro que nós definiremos)
        # Se verdadeiro, adicionamos a transição de estado à memória
        # Se falso, colocamos a transição de estado em uma posição diferente
        if len(self.memoria) < self.capacidade:
            self.memoria.append(transicao_estado)
        else:
            self.memoria[self.posicao] = transicao_estado
            self.posicao = (self.posicao + 1) % self.capacidade

    # Método para geração de amostras
    # Retorna tensores para o treinamento do modelo
    def amostra(self, batchsize=10):

        # Vamos trabalhar com batches de dados para a memória
        minibatch = random.sample(self.memoria, batchsize)

        # Extraímos as amostras ara cada componente
        estado_batch = torch.stack(tuple(sample[0] for sample in minibatch))
        acao_batch = torch.stack(tuple(sample[1] for sample in minibatch))
        recompensa_batch = torch.stack(
            tuple(sample[2] for sample in minibatch))
        estado_1_batch = torch.stack(tuple(sample[3] for sample in minibatch))
        done_batch = torch.stack(tuple(sample[4] for sample in minibatch))

        return estado_batch, acao_batch, recompensa_batch, estado_1_batch, done_batch

    # Método para calcular o comprimento da memória
    def __len__(self):
        return len(self.memoria)
