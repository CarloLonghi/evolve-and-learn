import torch
from network import ActorCritic
from config import *


def transfer_weights(net_dest: ActorCritic):
    num_hinges = 8

    checkpoint = torch.load("data/PPO_580/spider/database1/last_checkpoint.zip")
    actor_critic = ActorCritic((num_hinges*NUM_OBS_TIMES, 4,), num_hinges)
    actor_critic.load_state_dict(checkpoint['model_state'])
    pi_or_encoder = actor_critic.actor.pi_encoder.encoders[1].state_dict()
    pi_final_encoder = actor_critic.actor.pi_encoder.final_encoder.state_dict()
    val_or_encoder = actor_critic.critic.val_encoder.encoders[1].state_dict()
    val_final_encoder = actor_critic.critic.val_encoder.final_encoder.state_dict()
    critic_layer = actor_critic.critic.critic_layer.state_dict()

    net_dest.actor.pi_encoder.encoders[1].load_state_dict(pi_or_encoder)
    net_dest.actor.pi_encoder.final_encoder.load_state_dict(pi_final_encoder)
    net_dest.critic.val_encoder.encoders[1].load_state_dict(val_or_encoder)
    net_dest.critic.val_encoder.final_encoder.load_state_dict(val_final_encoder)
    net_dest.critic.critic_layer.load_state_dict(critic_layer)

    return net_dest