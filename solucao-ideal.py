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

# Solução ideal
solucao_ideal = functions_blackjack.joga_blackjack_ideal_sol()

# Mostra as ações ideais executadas pelo Agente em cada estado
functions_blackjack.mostra_acoes_agente(ModeloDQN.ModeloDQN(solucao_ideal).y_hard_hand,
                                        ModeloDQN.ModeloDQN(
                                            solucao_ideal).y_soft_hand,
                                        title_hard='\nHard Hand\n',
                                        title_soft='\nSoft Hand\n',
                                        fig_title="\nSolução Ideal\n")
