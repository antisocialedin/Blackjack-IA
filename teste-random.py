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

# Loop  de 5 episódios do jogo para a política randômica
for episode in range(5):
    
    # Reset do ambiente
    env = gym.make('Blackjack-v1')
    estado = env.reset()
    
    # Enquanto for verdadeiro (enquanto o agente estiver jogando)
    while True:
        
        # Imprime o estado
        print('Estado:')
        print(estado)
        
        # Toma uma ação de forma aleatória a partir do espaço de ações disponíveis no ambiente
        action = env.action_space.sample()
        
        # Coletamos o resultado de um passo (ou uma jogada) 
        # Temos o estado, a recompensa, se o jogo terminou e informações
        estado, recompensa, done, info = env.step(action)
        
        # Se o jogo terminou
        if done:
            
            # Imprime a recompensa do agente
            print('\nO Jogo Terminou! Sua Recompensa: ', recompensa)
            
            # Imprime se agente venceu ou perdeu
            print('Você Venceu :)\n') if recompensa > 0 else print('Você Perdeu :(\n')
            
            # Encerra o loop neste episódio
            break