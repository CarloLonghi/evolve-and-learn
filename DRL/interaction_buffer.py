import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler
from config import NUM_OBSERVATIONS, NUM_STEPS, BATCH_SIZE, NUM_OBS_TIMES, PPO_GAMMA, PPO_LAMBDA

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

        self.actions = torch.zeros(NUM_STEPS, num_agents, act_dim)
        self.values = torch.zeros(NUM_STEPS + 1, num_agents)
        self.rewards = torch.zeros(NUM_STEPS, num_agents)
        self.logps = torch.zeros(NUM_STEPS, num_agents)
        self.advantages = torch.zeros(NUM_STEPS, num_agents)
        self.returns = torch.zeros(NUM_STEPS, num_agents)

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.step = 0

    def insert(self, obs, act, logp, val, rew):
        """
        Insert a new step in the replay buffer
        Used when inserting the data of a new step for multiple agents
        args:
            obs: observation at the current state
            act: action performed at the state
            logp: log probability of the action performed according to the current policy
            val: value of the state
            rew: reward received for performing the action
        """
        for i, observation in enumerate(obs):
            self.observations[i][self.step] = observation
        self.actions[self.step] = torch.tensor(act)
        self.values[self.step] = torch.tensor(val)[:,0]
        self.rewards[self.step] = torch.tensor(rew)
        self.logps[self.step] = torch.tensor(logp)

        self.step += 1

    def insert_single(self, idx, obs, act, logp, val, rew):
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
        """
        for i, observation in enumerate(obs):
            self.observations[i][self.step, idx] = torch.from_numpy(observation)
        self.actions[self.step, idx] = torch.tensor(act)
        self.values[self.step, idx] = torch.tensor(val)
        self.rewards[self.step, idx] = rew
        self.logps[self.step, idx] = torch.tensor(logp)

        self.step += 1

    def set_last_value(self, next_state_value):
        """
        Insert the value of the last state reached
        """
        self.values[-1] = next_state_value

    def set_single_last_value(self, idx, next_state_value):
        """
        Insert the value of the last state reached
        """
        self.values[-1, idx] = next_state_value

    def _compute_advantages(self):
        """
        Compute the advantage function and the returns used to compute the loss
        """
        adv = 0
        for t in range((NUM_STEPS -1), -1, -1):
            delta = self.rewards[t] + PPO_GAMMA * self.values[t+1] - self.values[t]
            adv = delta + (PPO_LAMBDA * PPO_GAMMA) * adv
            self.advantages[t] = adv
            self.returns[t] = adv + self.values[t]

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
            val = self.values.view(-1)[idxs]
            act = self.actions.view(-1, self.act_dim)[idxs]
            logp_old = self.logps.view(-1)[idxs]
            rew = self.rewards.view(-1)[idxs]
            adv = self.advantages.view(-1)[idxs]
            ret = self.returns.view(-1)[idxs]
            yield obs, val, act, logp_old, rew, adv, ret

    def reset_step_count(self):
        self.step = 0
