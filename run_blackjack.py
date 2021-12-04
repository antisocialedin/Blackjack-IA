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

import functions_blackjack
import ModeloDQN
import treinamento

# Solução ideal
solucao_ideal = functions_blackjack.joga_blackjack_ideal_sol()

# Mostra as ações ideais executadas pelo Agente em cada estado
functions_blackjack.mostra_acoes_agente(ModeloDQN.ModeloDQN(solucao_ideal).y_hard_hand,
                                        ModeloDQN.ModeloDQN(
                                            solucao_ideal).y_soft_hand,
                                        title_hard='\nHard Hand\n',
                                        title_soft='\nSoft Hand\n',
                                        fig_title="\nSolução Ideal\n")

# Definimos se usaremos GPU ou CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Criamos o modelo
modelo = ModeloDQN.ModeloDQN(solucao_ideal).to(device)

""" # Verificamos se a execução é no Google Colab
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import google.colab
    from google.colab import drive

# Preparamos o caminho para salvar o modelo
if IN_COLAB:
    drive.mount('/content/drive')
    ROOT = "/content/drive/My Drive/DeepLearningBook/"
    sys.path.append(ROOT)
else:
    ROOT = r""
    sys.path.append(ROOT) """

# save_dir = ROOT + 'Modelos'
save_dir = './loggers'

# Nome e versão do modelo a ser salvo
version = 1.0
model_name = f'Modelo_Blackjack_{version}'
save_path = os.path.join(save_dir, model_name + ".pt")

# Treina o modelo
loggers = treinamento.treina_modelo(modelo, save_path)

# Avalia o modelo
hard_final, soft_final = treinamento.avalia_modelo(modelo, display_result = False, episode_idx = treinamento.TOTAL_EPISODES)
print(f'Hard Acurácia Final: {hard_final:0.2f}, Soft Acurácia Final: {soft_final:0.2f}')