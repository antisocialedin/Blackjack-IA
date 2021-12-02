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

# Definindo as variáveis de controle

# Número de rounds
num_rounds = 1000 

# Números de amostras
num_samples = 100 

# Lista para armazenar a recompensa média
average_payouts = []

# Gravamos o horário que começou o treinamento
start_time = time.time()

# Loop pelo range de amostras
for sample in range(num_samples):
    
    # Reset do ambiente
    env = gym.make('Blackjack-v1')

    # Inicialia o round
    round = 1
    
    # Recompensa total
    total_payout = 0 
    
    # Loop pelo número de rounds
    while round <= num_rounds:
        
        # Toma ação aleatória
        action = env.action_space.sample()  
        
        # Extrai o resultado de cada passada (cada jogada)
        # Como não queremos info, colocamos apenas _
        obs, payout, is_done, _ = env.step(action)
        
        # Totaliza a recompensa
        total_payout += payout
        
        # Verifica se o jogo terminou
        if is_done:
            
            # Reset do ambiente
            # O ambiente negocia novas cartas para o jogador e o dealer
            env.reset() 
            
            # Totaliza o round
            round += 1
    
    # Armazena as recompensas
    average_payouts.append(total_payout)