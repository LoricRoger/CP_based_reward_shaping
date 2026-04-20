# FL/config.py

seed_value = 1

# Q-learning hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.995  # Gamma
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.99  # Epsilon decay rate per episode
EPSILON_MIN = 0.01  # Minimum exploration rate

# Q-table initialization values
Q_INIT_VALUE_NONE = 0.0
Q_INIT_VALUE_CLASSIC = 0.0
Q_INIT_VALUE_CP_MS = 0.0
Q_INIT_VALUE_CP_ETR = 0.0
Q_INIT_VALUE_CP_ETR_BUDGET_NOSLIP = -0.1  # Valeur initiale pour les actions no-slip (actions 4-7) quand budget > 0

# Reward shaping coefficients
CP_MS_SHAPING_COEFF = 0.2   # Coefficient appliqué à (marginal - 0.25) en mode CP-MS
NOSLIP_PENALTY_COEFF = 0.05  # Pénalité par action no-slip utilisée (stratégie 'penalize')

# Evaluation parameters
EVAL_EPISODES = 100  # Number of episodes *per evaluation run*
EVAL_FREQUENCY = 10  # Evaluate the agent every X training episodes
