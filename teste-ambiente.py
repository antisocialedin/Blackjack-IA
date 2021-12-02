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
from IPython.display import clear_output

# Cria uma instância de um ambiente que facilita o desenvolvimento e teste do nosso agente
env = gym.make('Blackjack-v1')
env.reset()

# Extraímos env.action_space para verificar o espaço de ações disponíveis para o agente
print(f'Espaço de Ação: {env.action_space}. \n\n=> (1 = Hit, 0 = Bust)\n') 

# Extraímos env.observation_space para verificar o espaço de observação disponível para o agente
print(f'Espaço de Observação: {env.observation_space} \n\n=> (Total da Mão do Jogador, Carta do Dealer, Ás Utilizável)')

print("Espaço de Observação:")
print('=> ', env.step(0))