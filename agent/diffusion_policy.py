import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.ir_sde import IR_SDE
from agent.model import DiffusionMLP

class DiffusionPolicy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 sde=None,
                 n_timesteps=100,
                 loss_type='MLL',
                 predict_epsilon=True,
                 use_subgoal=False,
                 subgoal_dim=0
                ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon
        self.loss_type = loss_type
        
        self.use_subgoal = use_subgoal
        input_dim = state_dim + (subgoal_dim if use_subgoal else 0)

        self.model = DiffusionMLP(state_dim=input_dim, action_dim=action_dim)
        self.sde = sde

    def clip_action(self, action):
        return action.clamp(-self.max_action, self.max_action)

    def predict_start_from_score(self, a_t, t, score):
        action = self.sde.predict_start_from_score(a_t, t, score)
        return action

    def previous_diffusion_action(self, a_t, t=0):
        # action = self.sde.forward_step(a_t, t)
        action = a_t + self.sde.drift(a_t, t) * self.sde.dt
        return action

    def sample_noise(self, tensor):
        batch_size = tensor.shape[0]
        return torch.randn(batch_size, self.action_dim).to(tensor.device)

    def sample_action(self, state, mode):
        score_fn = self.model if mode == 'posterior' else self.score_fn
        noise = self.sample_noise(state)
        action = self.sde.reverse(noise, score_fn, mode=mode, clip_value=self.max_action, state=state)
        return self.clip_action(action)

    # compute ood
    def score_fn(self, a_t, t, state):
        noise = self.model(a_t, t, state=state)
        return self.sde.compute_score_from_noise(noise, t)
    
    def random_action_states(self, a_0):
        a_t, t, noise = self.sde.generate_random_states(a_0)
        return a_t, t, noise

    # noise matching loss
    def NML_loss(self, state, a_0):
        a_t, t, noise = self.random_action_states(a_0)
        noise_pred = self.model(a_t, t.squeeze(1), state)
        start_pred = self.sde.predict_start_from_noise(a_t, t, noise_pred)
        
        return F.mse_loss(noise_pred, noise), start_pred

    # maximum likelihood loss
    def MLL_loss(self, state, a_0):
        a_t, t, noise = self.random_action_states(a_0)
        noise_pred = self.model(a_t, t.squeeze(1), state)
        score_pred = self.sde.compute_score_from_noise(noise_pred, t)
        start_pred = self.sde.predict_start_from_noise(a_t, t, noise_pred)

        a_pre_pred = a_t - self.sde.sde_reverse_drift(a_t, t, score_pred) * self.sde.dt
        a_pre_target = self.sde.reverse_optimum_step(a_t, a_0, t)
        return F.mse_loss(a_pre_pred, a_pre_target), start_pred

    def loss(self, state, a_0, subgoal=None):
        if self.use_subgoal and subgoal is not None:
            net_in = torch.cat([state, subgoal], dim=-1)
        else:
            net_in = state
        
        if self.loss_type == 'MLL':
            return self.MLL_loss(net_in, a_0)
        elif self.loss_type == 'NML':
            return self.NML_loss(net_in, a_0)
        else:
            print('Now only support MLL and NML loss')

    def forward(self, state, subgoal=None, mode='posterior'):
        if self.use_subgoal and subgoal is not None:
            net_in = torch.cat([state, subgoal], dim=-1)
        else:
            net_in = state
        return self.sample_action(net_in, mode=mode)