
# simulation parameters
SAMPLING_FREQUENCY = 4
CONTROL_FREQUENCY = 4
NUM_ITERATIONS = 30
NUM_STEPS = 128
SIMULATION_TIME = NUM_ITERATIONS*(NUM_STEPS/CONTROL_FREQUENCY)

# PPO parameters
PPO_CLIP_EPS = 0.2
PPO_LAMBDA = 0.95
PPO_GAMMA = 0.99

# loss weights
ACTOR_LOSS_COEFF = 1
CRITIC_LOSS_COEFF = 1
ENTROPY_COEFF = 0.01

# learning rates
LR_ACTOR = 8e-4
LR_CRITIC = 1e-3

BATCH_SIZE = 256
N_EPOCHS = 4

NUM_PARALLEL_AGENT = 8

# number of past hinges positions to pass as observations
NUM_OBS_TIMES = 3

# dimension of the different types of  observations
NUM_OBSERVATIONS = 2