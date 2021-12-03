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

# Classe para o modelo Deep Q-Network


class ModeloDQN(nn.Module):

    # Método construtor da classe
    def __init__(self,
                 ideal_sol,
                 number_of_states=3,
                 number_of_actions=2,
                 weight_norm=False):

        super(ModeloDQN, self).__init__()

        # Inicializa os atributos do modelo
        self.current_episode = 0
        self.fc1 = nn.Linear(number_of_states, 96)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(96, 48)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(48, 24)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(24, number_of_actions)

        # Normaliza os pesos
        if weight_norm:
            self.fc1 = nn.utils.weight_norm(self.fc1)
            self.fc2 = nn.utils.weight_norm(self.fc2)
            self.fc3 = nn.utils.weight_norm(self.fc3)

        # Aplica a solução ideal
        self.x_soft_hand = ideal_sol[0]
        self.x_hard_hand = ideal_sol[1]
        self.y_soft_hand = ideal_sol[2]
        self.y_hard_hand = ideal_sol[3]

    # Método com a passada para frente (forward)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)

        return out
