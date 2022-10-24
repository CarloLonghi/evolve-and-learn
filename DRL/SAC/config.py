
# simulation parameters
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 5
NUM_ITERATIONS = 28
NUM_STEPS = 150
SIMULATION_TIME = 30

# SAC parameters
INIT_TEMPERATURE = 0.1

# loss weights
ACTOR_LOSS_COEFF = 1
CRITIC_LOSS_COEFF = 1

# learning rates
LR_ACTOR = 8e-4
LR_CRITIC = 1e-3
LR_ALPHA = 1e-4

# other parameters
GAMMA = 0.99

BATCH_SIZE = 450
N_EPOCHS = 4

NUM_PARALLEL_AGENT = 10

# number of past hinges positions to pass as observations
NUM_OBS_TIMES = 3

# dimension of the different types of  observations
NUM_OBSERVATIONS = 2

ACTION_CONSTRAINT = 1 #[-ACTION_CONSTRAINT, ACTION_CONSTRAINT]