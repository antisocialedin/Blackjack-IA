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

# Função para representar as jogadas do Agente de forma visual


def cria_heatmap(data,
                 row_labels,
                 col_labels,
                 ax=None,
                 cbar_kw={},
                 cbarlabel="",
                 **kwargs):

    # Verifica o eixo
    if not ax:
        ax = plt.gca()

    # Prepara o que será mostrado no eixo
    im = ax.imshow(data, **kwargs)

    # Ajusta os limites
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Posição
    ax.tick_params(top=True,
                   bottom=False,
                   labeltop=True,
                   labelbottom=False)

    # Gira os rótulos dos marcadores e define seu alinhamento.
    plt.setp(ax.get_xticklabels(),
             rotation=-30,
             ha="right",
             rotation_mode="anchor")

    # Cria uma grade branca.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # Limites do eixo x
    ax.set_xticks(np.arange(data.shape[1]+1)-.5,
                  minor=True)

    # Limites do eixo y
    ax.set_yticks(np.arange(data.shape[0]+1)-.5,
                  minor=True)

    # Grid
    ax.grid(which="minor",
            color="w",
            linestyle='-',
            linewidth=4)

    # Parâmetros
    ax.tick_params(which="minor",
                   bottom=False,
                   left=False)

    return im

# Função para preencher o Heatmap


def grava_texto_heatmap(im,
                        data=None,
                        valfmt="{x:.2f}",
                        textcolors=["black", "white"],
                        threshold=None,
                        **textkw):

    # Obtém os dados para preencher o heatmap
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Verifica os limites
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Dicionário de alinhamento
    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    kw.update(textkw)

    # Formatação
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Cria uma lista para os textos que serão colocados no heatmap (0 ou 1, como resultado da escolha do Agente)
    texts = []

    # Loop pelos dados e preenchimento da lista
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# Função para criar a solução ideal


def joga_blackjack_ideal_sol():

    # Inicializa os estados do dealer e do Agente para as jogadas mais difíceis
    dealer_states = 10
    player_states = 18

    # Aqui faremos as jogadas mais difíceis
    # Agente 4 - 21
    # Dealer 2 - 11
    x_hard_hand = np.zeros((player_states * dealer_states, 3))
    for i in range(player_states * dealer_states):
        dealer_card = np.floor(i / player_states)
        player_card = i % player_states
        x_hard_hand[i] = (player_card + 4, dealer_card + 2, 0)

    # E aqui a solução ideal para as jogadas mais difíceis
    y_hard_hand = np.ones((player_states, dealer_states))
    for player_total in range(9, player_states + 4):
        for dealer_total in range(2, dealer_states + 2):
            player_idx = player_total - 4
            dealer_idx = dealer_total - 2

            # Checa as regras do jogo
            if player_total >= 17:
                y_hard_hand[player_idx][dealer_idx] = 0

            if 13 <= player_total <= 16:
                if dealer_total <= 6:
                    y_hard_hand[player_idx][dealer_idx] = 0

            if player_total == 12 and 4 <= dealer_total <= 6:
                y_hard_hand[player_idx][dealer_idx] = 0

    # Inicializa os estados do dealer e do Agente para as jogadas mais fáceis
    dealer_states = 10
    player_states = 8

    # Aqui faremos as jogadas mais fáceis
    # Agente 13 - 20
    # Dealer 2 - 11
    x_soft_hand = np.zeros((player_states * dealer_states, 3))
    for i in range(player_states * dealer_states):
        dealer_card = np.floor(i / player_states)
        player_card = i % player_states
        x_soft_hand[i] = (player_card + 13, dealer_card + 2, 1)

    # E aqui a solução ideal para as jogadas mais fáceis
    y_soft_hand = np.ones((player_states, dealer_states))
    for player_total in range(18, player_states + 13):
        for dealer_total in range(2, dealer_states + 2):
            player_idx = player_total - 13
            dealer_idx = dealer_total - 2

            # Checa as regras do jogo
            if player_total >= 19:
                y_soft_hand[player_idx][dealer_idx] = 0

            if player_total == 18:
                if dealer_total <= 8:
                    y_soft_hand[player_idx][dealer_idx] = 0

    # Convertemos os resultados em tensores PyTorch para o processamento do modelo
    x_soft_hand = torch.from_numpy(x_soft_hand).float()
    x_hard_hand = torch.from_numpy(x_hard_hand).float()
    y_soft_hand = torch.from_numpy(y_soft_hand.reshape((-1, 1))).float()
    y_hard_hand = torch.from_numpy(y_hard_hand.reshape((-1, 1))).float()

    # Retorno da função
    return x_soft_hand, x_hard_hand, y_soft_hand, y_hard_hand

# Função pra mostrar as ações do Agente em cada estado possível


def mostra_acoes_agente(hard_preds,
                        soft_preds,
                        title_hard='',
                        title_soft='',
                        fig_title=''):

    # Estados possíveis do dealer
    dealer_states = list(range(2, 12))

    # Estados possíveis do jogador (Agente)
    player_hard_states = list(range(4, 22))
    player_soft_states = ['A, 2', 'A, 3', 'A, 4',
                          'A, 5', 'A, 6', 'A, 7', 'A, 8', 'A, 9', 'A, 10']

    # Ações possíveis
    hard_hand_actions = hard_preds.int().numpy().reshape((18, 10))
    soft_hand_actions = soft_preds.int().numpy().reshape((8, 10))

    # Cria a área para o plot e subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Grava o título
    ax[0].title.set_text(title_hard)

    # Cria o heatmap - Hard
    im = cria_heatmap(hard_hand_actions,
                      player_hard_states,
                      dealer_states,
                      ax=ax[0],
                      cmap="Purples")

    # Grava no heatmap as jogadas do Agente
    grava_texto_heatmap(im,
                        valfmt="{x}")

    # # Cria o heatmap - Soft
    im = cria_heatmap(soft_hand_actions,
                      player_soft_states,
                      dealer_states,
                      ax=ax[1],
                      cmap="Purples")

    # Grava no heatmap as jogadas do Agente
    grava_texto_heatmap(im,
                        valfmt="{x}")

    # Grava o título
    ax[1].title.set_text(title_soft)

    # Título do Plot
    if fig_title:
        fig.suptitle(fig_title)

    fig.tight_layout()
    plt.show()

