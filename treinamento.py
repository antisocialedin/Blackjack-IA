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
import memoria_agente

# Hiperparâmetros para treinamento do Modelo
NUM_ACOES = 2
NUM_ESTADOS = 3
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.01
EPSILON_DECAY = 0.9  
GAMMA = 0.999  
REPLAY_MEMORY_SIZE = 5000
MINIBATCH_SIZE = 256
STEPS_PER_EPISODE = 100
TOTAL_EPISODES = 50_000
LEARNING_RATE = 1e-5

# Dicionário de hiperparâmetros
hyperparams_dict = {'NUM_ACOES': NUM_ACOES,
                    'NUM_ESTADOS': NUM_ESTADOS,
                    'EPSILON_FINAL': EPSILON_FINAL,
                    'EPSILON_DECAY': EPSILON_DECAY,
                    'GAMMA': GAMMA,
                    'REPLAY_MEMORY_SIZE': REPLAY_MEMORY_SIZE,
                    'MINIBATCH_SIZE': MINIBATCH_SIZE,
                    'STEPS_PER_EPISODE': STEPS_PER_EPISODE,
                    'TOTAL_EPISODES': TOTAL_EPISODES,
                    'LEARNING_RATE': LEARNING_RATE,
                   }

env = gym.make('Blackjack-v1')

# Definimos se usaremos GPU ou CPU

#Para Rodar na GPU e se "ausente" rodar na CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Para rodar na GPU
# device = torch.device('cuda') 

# Para rodar na CPU                                        
# device = torch.device('cpu')                                             

save_dir = './loggers'

# Função para o treinamento de um único episódio
def treinamento_episodio(model, optimizer, memory, loss_func, epsilon, n_steps, loss_logger):
    
    # Reset do estado
    cur_state = env.reset()
    
    # Treinamento
    with torch.no_grad():
        
        # Fazemos com que a modelo jogue blackjack por n_steps e salve o resultado na memória, 
        # usando o epsilon ganancioso para explorar o ambiente
        for step_i in range(n_steps):
            cur_state_t = torch.Tensor(cur_state).to(device)
            action = obtem_acao(model, cur_state_t, epsilon)
            next_state, reward, is_done, _ = env.step(action)
            memory.carrega_memoria(cur_state, action, next_state, reward, is_done)
            
            # Se concluído, fazemos o reset do ambiente, se não, avançamos ao próximo estado
            if is_done:
                cur_state = env.reset()  
            else:
                cur_state = next_state

    # A cada n_steps, atualizamos o modelo com base nos loggers
    if len(memory) > MINIBATCH_SIZE:
        atualiza_modelo(model, optimizer, memory, loss_func, loss_logger)

    return loss_logger

# Função para treinamento do agente por milhares de episódios (rodadas de Blackjack)
def treinamento_agente(model, 
                       optimizer, 
                       memory, 
                       loss_func, 
                       n_steps, 
                       n_episodes, 
                       save_path, 
                       loggers, 
                       exp_epsilon_decay = False):
    
    # Preparamos os logs para monitorar a evolução do treinamento
    training_loss_logger, hard_accuracy_logger, soft_accuracy_logger = loggers

    # Lista para registrar os decrementos do valor de epsilon
    epsilon_decrements = []
    
    # Condicional
    if exp_epsilon_decay:
        epsilon_decrements = [EPSILON_INITIAL]
        found_eps_min = False
        
        for i in range(TOTAL_EPISODES):
            
            if epsilon_decrements[i] > EPSILON_FINAL:
                epsilon_decrements.append(epsilon_decrements[i] * EPSILON_DECAY)
            elif not found_eps_min:
                epsilon_decrements.append(epsilon_decrements[i])
                print(f'Valor Mínimo de Epsilon alcançado em {i} episódios')
                found_eps_min = True
            else:
                epsilon_decrements.append(epsilon_decrements[i])
   
    else:
        epsilon_decrements = np.linspace(EPSILON_INITIAL, EPSILON_FINAL, n_episodes+1)

    # Registra o início do treinamento
    start_time = time.time()
    
    # Loop por cada episódio
    for episode_idx in range(model.current_episode, model.current_episode + n_episodes + 1):
        
        # Decremento do epsilon
        epsilon = epsilon_decrements[episode_idx - model.current_episode] 

        # Chamada à função de treinamento
        training_loss_logger = treinamento_episodio(model, 
                                                    optimizer, 
                                                    memory, 
                                                    loss_func, 
                                                    epsilon, 
                                                    n_steps, 
                                                    training_loss_logger)

        # Registra a acurácia do modelo no episódio
        hard_accuracy, soft_accuracy = avalia_modelo(model)
        hard_accuracy_logger.append(hard_accuracy)
        soft_accuracy_logger.append(soft_accuracy)
        loggers = training_loss_logger, hard_accuracy_logger, soft_accuracy_logger

        # Salva o checkpoint a cada 2000 episódios
        if episode_idx % 2000 == 0:
            avalia_modelo(model, display_result = True, episode_idx = episode_idx)
            torch.save({'episode_idx': episode_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loggers': loggers,
                        'hyperparams_dict': hyperparams_dict,
                        'episode_idx': episode_idx}, 
                       save_path)
            print('Modelo Salvo')

        if episode_idx % 200 == 0:
            print(f'| Episódio: {episode_idx:02} | Erro em Treinamento: {training_loss_logger[episode_idx-1]:0.2f} '
                  f'| Acurácia Hard: {hard_accuracy:0.2f} | Acurácia Soft: {soft_accuracy:0.2f} |')

    end_time = time.time()
    print("Tempo de Total de Treinamento igual a %.2f segundos" % (end_time - start_time))

    return loggers

# Função para avaliar o modelo
def avalia_modelo(model, display_result = False, episode_idx = 0):
    
    model.eval()
    
    with torch.no_grad():
        
        x_hard_hand = model.x_hard_hand.to(device)  
        x_soft_hand = model.x_soft_hand.to(device)  


        y_pred_hard = model(x_hard_hand)
        y_pred_hard = torch.max(y_pred_hard, 1)[1].float()
        y_pred_hard = y_pred_hard.reshape((10, 18)).t().reshape((-1, 180)).squeeze()

        y_pred_soft = model(x_soft_hand)
        y_pred_soft = torch.max(y_pred_soft, 1)[1].float()  
        y_pred_soft = y_pred_soft.reshape((10, 8)).t().reshape((-1, 80)).squeeze()

        y_hard = model.y_hard_hand.squeeze()
        y_soft = model.y_soft_hand.squeeze()

        correct_action_hard = torch.eq(y_hard, y_pred_hard)
        correct_action_soft = torch.eq(y_soft, y_pred_soft)

        correct_action_hard = correct_action_hard.reshape((18, 10))
        correct_action_soft = correct_action_soft.reshape((8, 10))

        fig_title = f'Episódios: {episode_idx}' if episode_idx > 0 else None

        if display_result:
            functions_blackjack.mostra_acoes_agente(y_pred_hard, y_pred_soft, 
                               "Ações de Hard Hand do Modelo", 
                               "Ações de Soft Hand do Modelo", 
                               fig_title = fig_title)

        if correct_action_hard.sum().item() != 0:
            hard_hand_accuracy =  correct_action_hard.sum().item() / correct_action_hard.numel() * 100
        else:
            hard_hand_accuracy = 0
        if correct_action_soft.sum().item() != 0:
            soft_hand_accuracy = correct_action_soft.sum().item() / correct_action_soft.numel() * 100
        else:
            soft_hand_accuracy = 0

        return hard_hand_accuracy, soft_hand_accuracy

# Função para obter uma ação
def obtem_acao(model, cur_state, epsilon):

    # Verifica o hiperparâmetro epsilon para decidir se retornamos uma ação aleatória ou uma ação prevista
    if np.random.rand() < epsilon:
        return np.random.randint(0, NUM_ACOES)
    else:
        acao_prevista = model(cur_state)
        values, index = acao_prevista.max(0)

        return index.item()

# Atualiza a rede
def atualiza_modelo(model, optimizer, memory, loss_func, loss_logger):

    # Coloca o modelo em modo de treinamento
    model.train()
    
    # Executa ações
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.amostra(MINIBATCH_SIZE)
    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    done_batch = done_batch.to(device)
    current_state_qs = model(state_batch) 
    next_state_qs = model(next_state_batch)
    target_qs_list = []

    # Loop pelos mini-batches
    for i in range(MINIBATCH_SIZE):
        if not done_batch[i]:
            max_future_q = torch.max(next_state_qs[i])
            new_q = reward_batch[i] + GAMMA * max_future_q
        else:
            new_q = reward_batch[i]

        action_taken = int(action_batch[i].item())
        current_qs = current_state_qs[i].clone() 
        current_qs[action_taken] = new_q
        target_qs_list.append(current_qs)

    target_q_values = torch.stack(target_qs_list)

    optimizer.zero_grad()

    # Calcula o erro
    loss = loss_func(current_state_qs, target_q_values)
    loss_logger.append(loss.item())
    loss.backward()

    # Atualiza o modelo
    optimizer.step()
    model.eval()

# Função para carregar o modelo
def carrega_modelo(model, 
                   save_dir, 
                   save_path, 
                   load_checkpoint = False, 
                   load_hyperparams = False):
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if load_checkpoint:
        
        if os.path.isfile(save_path):

            check_point = torch.load(save_path)
            model.load_state_dict(check_point['model_state_dict'])
            model.optimizer.load_state_dict(check_point['optimizer_state_dict'])
            model.current_episode = check_point['episode_idx']
            model.loggers = check_point['loggers']
            print(f'Hard Accuracy Corrente: {model.loggers[1][model.current_episode]:0.2f}, '
                  f'Soft Accuracy Corrente: {model.loggers[2][model.current_episode]:0.2f}')
            print("Checkpoint Carregado. Iniciando do episódio:", model.current_episode)
            return True
        
        else:
            print("Checkpoint não encontrado!")
            return False

    elif load_hyperparams:
        check_point = torch.load(save_path)
        hyperparams_dict = check_point['hyperparams_dict']

        global NUM_ACOES, NUM_ESTADOS, EPSILON_FINAL, EPSILON_FINAL, EPSILON_DECAY, GAMMA, \
               REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, STEPS_PER_EPISODE, TOTAL_EPISODES, LEARNING_RATE
        
        NUM_ACOES = hyperparams_dict['NUM_ACOES']
        NUM_ESTADOS = hyperparams_dict['NUM_ESTADOS']
        EPSILON_FINAL = ['EPSILON_FINAL']
        EPSILON_DECAY = hyperparams_dict['EPSILON_DECAY']
        GAMMA = hyperparams_dict['GAMMA']
        REPLAY_MEMORY_SIZE = hyperparams_dict['REPLAY_MEMORY_SIZE']
        MINIBATCH_SIZE = hyperparams_dict['MINIBATCH_SIZE']
        STEPS_PER_EPISODE = hyperparams_dict['STEPS_PER_EPISODE']
        TOTAL_EPISODES = hyperparams_dict['TOTAL_EPISODES']
        LEARNING_RATE = hyperparams_dict['LEARNING_RATE']

        print('Hiperparâmetros Carregados do Checkpoint!')
        return False

    else:
        print('Modelo Não Carregado!')
        return False

# Função para treinar o modelo
def treina_modelo(model, save_path):
    
    # Otimizador
    model.optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    # Memória
    memory = memoria_agente.Memoria(REPLAY_MEMORY_SIZE)
    
    # Loss
    loss_func = nn.MSELoss()

    # Reset dos loggers
    training_loss_logger = [0]
    hard_accuracy_logger = []
    soft_accuracy_logger = []
    model.loggers = training_loss_logger, hard_accuracy_logger, soft_accuracy_logger

    # Carrega a versão anterior do modelo
    load_checkpoint = True
    load_only_hyperparams = False
    checkpoint_loaded = carrega_modelo(model, 
                                       save_dir, 
                                       save_path, 
                                       load_checkpoint, 
                                       load_only_hyperparams)

    if not checkpoint_loaded:
        loggers = treinamento_agente(model, 
                                     model.optimizer, 
                                     memory, 
                                     loss_func, 
                                     STEPS_PER_EPISODE, 
                                     TOTAL_EPISODES, 
                                     save_path, 
                                     model.loggers)
    
    return model.loggers