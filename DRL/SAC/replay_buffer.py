import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler
from config import NUM_OBSERVATIONS, NUM_STEPS, BATCH_SIZE

class Buffer(object):
    """
    Replay buffer
    It stores observations, actions, values and rewards for each step of the simulation
    Used to create batches of data to train the controller 
    """
    def __init__(self, obs_dim, act_dim, num_agents):
        self.observations = []
        for dim in obs_dim:
            self.observations.append(torch.zeros(NUM_STEPS, num_agents, dim))

        self.next_observations = []
        for dim in obs_dim:
            self.next_observations.append(torch.zeros(NUM_STEPS, num_agents, dim))

        self.actions = torch.zeros(NUM_STEPS, num_agents, act_dim)
        self.rewards = torch.zeros(NUM_STEPS, num_agents)
        self.logps = torch.zeros(NUM_STEPS, num_agents)
        self.last_state_value = torch.zeros(num_agents)

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.step = 0

    def insert(self, obs, act, logp, rew, next_obs):
        """
        Insert a new step in the replay buffer
        Used when inserting the data of a new step for multiple agents
        args:
            obs: observation at the current state
            act: action performed at the state
            logp: log probability of the action performed according to the current policy
            val: value of the state
            rew: reward received for performing the action
            next_obs: observation at the next state
        """
        for i, observation in enumerate(obs):
            self.observations[i][self.step] = observation
        self.actions[self.step] = torch.tensor(act)
        self.rewards[self.step] = torch.tensor(rew)
        self.logps[self.step] = torch.tensor(logp)
        for i, observation in enumerate(next_obs):
            self.next_observations[i][self.step] = observation

        self.step += 1

    def insert_single(self, idx, obs, act, logp, rew, next_obs):
        """
        Insert a new step in the replay buffer
        Used when inserting the data of a new step for a single agent
        args:
            idx: the index of the agent to which the data is referred
            obs: observation at the current state
            act: action performed at the state
            logp: log probability of the action performed according to the current policy
            val: value of the state
            rew: reward received for performing the action
            next_obs: observation at the next state
        """
        for i, observation in enumerate(obs):
            self.observations[i][self.step, idx] = torch.from_numpy(observation)
        self.actions[self.step, idx] = torch.tensor(act)
        self.rewards[self.step, idx] = rew
        self.logps[self.step, idx] = torch.tensor(logp)
        for i, observation in enumerate(next_obs):
            self.next_observations[i][self.step, idx] = torch.from_numpy(observation)

        self.step += 1

    def _normalize_rewards(self):
        """
        Normalize the rewards obtained
        """
        rewards = self.rewards.view(-1)
        min = rewards.min()
        self.rewards = self.rewards - min
        max = rewards.max()
        self.rewards = self.rewards / max

    def get_sampler(self,):
        """
        Create a BatchSampler that divides the data in the buffer in batches 
        """
        dset_size = NUM_STEPS * self.num_agents
        batch_size = BATCH_SIZE

        assert dset_size >= batch_size

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            batch_size,
            drop_last=True,
        )

        for idxs in sampler:
            obs = [[] for _ in range(NUM_OBSERVATIONS)]
            for i, o in enumerate(self.observations):
                obs[i] = o.view(-1, self.obs_dim[i])[idxs]
            act = self.actions.view(-1, self.act_dim)[idxs]
            logp_old = self.logps.view(-1)[idxs]
            rew = self.rewards.view(-1)[idxs]
            next_obs = [[] for _ in range(NUM_OBSERVATIONS)]
            for i, o in enumerate(self.next_observations):
                next_obs[i] = o.view(-1, self.obs_dim[i])[idxs]
            yield obs, act, logp_old, rew, next_obs

    def reset_step_count(self):
        self.step = 0
