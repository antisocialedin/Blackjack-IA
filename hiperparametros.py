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