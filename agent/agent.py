import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agent.ir_sde import IR_SDE
from agent.diffusion_policy import DiffusionPolicy
from agent.model import EnsembleCritic, SGPolicy, SGCritic
from agent.dynamics import Dynamics

from agent.helpers import EMA
from tqdm import tqdm

import wandb


class DynamicDiffusionPolicy(object):
    def __init__(self,
            state_dim,
            action_dim,
            max_action,
            device,
            discount,
            tau,
            replay_buffer,
            dynamics = None,
            alpha = 1.0,
            do_pretrain = True,
            horizon = 5,      
            rollout_steps = 5,           
            subgoals_cand = 5,
            subgoal_reg = 0.1,        
            risk_threshold = 0.05,    
            reward_risk_tradeoff  = 1.0,     
            subgoal_gamma = 0.9,
            buffer_for_subgoal = None,
            max_q_backup = False,
            eta = 1.0,           
            beta_schedule = 'cosine',      
            n_timesteps = 5,             
            ema_decay = 0.995,         
            step_start_ema  = 1000,          
            update_ema_every= 5,             
            lr = 1e-4,          
            lr_decay = False,
            lr_maxt = 1000,
            grad_norm = 1.0,           
            ent_coef  = 0.2,           
            num_critics = 4,             
            pess_method = 'lcb',         
            lcb_coef = 2.,
            loss_type = 'NML',
            action_clip = False,
            ):
        
        #-----------------------------------------------------------------------------#
        # Diffusion Policy (Actor)
        #-----------------------------------------------------------------------------#
        self.sde = IR_SDE(n_timesteps, beta_schedule, action_clip, device=device)
        self.actor = DiffusionPolicy(state_dim=state_dim,
                                     action_dim=action_dim,
                                     max_action=max_action,
                                     sde=self.sde,
                                     n_timesteps=n_timesteps,
                                     loss_type=loss_type,
                                     use_subgoal=True,
                                     subgoal_dim=state_dim
                                    ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.lr_decay = lr_decay
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
        
        #-----------------------------------------------------------------------------#
        # EMA(Actor) Model
        #-----------------------------------------------------------------------------#
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        self.step = 0
        self.step_start_ema = step_start_ema
        
        #-----------------------------------------------------------------------------#
        # Critic & Target Critic
        #-----------------------------------------------------------------------------#
        self.num_critics = num_critics
        self.critic = EnsembleCritic(state_dim, action_dim, num_critics=num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        if lr_decay:
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)
        
        #-----------------------------------------------------------------------------#
        # Env Model
        #-----------------------------------------------------------------------------#
        if dynamics is None:
            self.dynamics = Dynamics(state_dim, action_dim, ensemble_size=5,
                                  device=device, alpha=alpha)
            if do_pretrain and (replay_buffer is not None):
                self.dynamics.train_dynamics(replay_buffer, epochs=50, batch_size=100)
        else:
            self.dynamics = dynamics
            
        #-----------------------------------------------------------------------------#
        # Subgoal Policy
        #-----------------------------------------------------------------------------#
        self.subgoal_dim = state_dim
        self.sg_policy = SGPolicy(state_dim, self.subgoal_dim).to(device)
        self.sg_critic = SGCritic(state_dim, self.subgoal_dim).to(device)
        self.sg_target = copy.deepcopy(self.sg_critic)
        
        self.sg_optimizer = torch.optim.Adam(self.sg_policy.parameters(), lr=3e-4)
        self.sg_critic_optimizer = torch.optim.Adam(self.sg_critic.parameters(), lr=1e-4)
        
        self.sg_gamma  = subgoal_gamma
        self.sg_buffer = buffer_for_subgoal

        #-----------------------------------------------------------------------------#
        # Hyperparameters
        #-----------------------------------------------------------------------------#
        self.grad_norm    = grad_norm
        self.pess_method  = pess_method
        self.lcb_coef     = lcb_coef
        self.state_dim    = state_dim
        self.max_action   = max_action
        self.action_dim   = action_dim
        self.discount     = discount
        self.tau          = tau
        self.eta          = eta
        self.device       = device
        self.max_q_backup = max_q_backup
        self.ent_coef     = torch.tensor(ent_coef).to(self.device)
        
        self.horizon               = horizon
        self.rollout_steps         = rollout_steps
        self.subgoal_reg           = subgoal_reg
        self.subgoals_cand         = subgoals_cand
        self.risk_threshold        = risk_threshold
        self.reward_risk_tradeoff  = reward_risk_tradeoff

    #-----------------------------------------------------------------------------#
    # EMA Model Update
    #-----------------------------------------------------------------------------#
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)
    
    #-----------------------------------------------------------------------------#
    # Generate Subgoal
    #-----------------------------------------------------------------------------#
    @torch.no_grad()
    def get_subgoal(self, state):
        subgoal=self.sg_policy(state)
        return subgoal
    
    def get_subgoal_candidates(self, state, subgoal, n_candidates=None):

        if n_candidates is None:
            n_candidates = self.subgoals_cand

        batch_size = state.shape[0]
        
        rollout_states    = []
        rollout_actions   = []
        rollout_penalties = []
        
        current_state = state.clone()
        
        for step_idx in range(self.rollout_steps):
            a_t = self.actor(current_state, subgoal)
            next_s, _, pen = self.dynamics.simulate_ensemble(current_state, a_t)
            rollout_states.append(next_s)
            rollout_actions.append(a_t)
            rollout_penalties.append(pen)
            current_state = next_s.clone()

        rollout_states    = torch.stack(rollout_states, dim=0)
        rollout_penalties = torch.stack(rollout_penalties, dim=0)
        rollout_actions   = torch.stack(rollout_actions, dim=0)

        rollout_states    = rollout_states.permute(1, 0, 2).contiguous()
        rollout_penalties = rollout_penalties.permute(1, 0, 2).contiguous()
        rollout_actions   = rollout_actions.permute(1, 0, 2).contiguous()
        
        flat_states = rollout_states.view(batch_size * self.rollout_steps, self.state_dim)
        flat_actions = rollout_actions.view(batch_size * self.rollout_steps, self.action_dim)
        q_ens = self.critic_target(flat_states, flat_actions)
        
        # compute pessimistic Q
        if self.pess_method == 'lcb':
            mu    = q_ens.mean(dim=1, keepdim=True)
            std   = q_ens.std(dim=1, keepdim=True)
            q_val = mu - self.lcb_coef * std
        else:
            q_val = q_ens.min(dim=1, keepdim=True)[0]
        
        q_val = q_val.view(batch_size, self.rollout_steps)
        penalties = rollout_penalties.view(batch_size, self.rollout_steps)
        
        # compute score
        score = q_val - self.reward_risk_tradeoff * penalties
        
        # masking
        valid_mask = (penalties < self.risk_threshold)
        score = torch.where(valid_mask, score, torch.tensor(-9999., device=self.device))
        
        # extract top-k on each batch
        topk_score, topk_idx = torch.topk(score, k=n_candidates, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, self.state_dim)
        
        best_subgoals = torch.gather(rollout_states, 1, gather_idx)
        return best_subgoals
    
    def save_sg_transition(self, state, goal, h_reward, next_state):
        if self.sg_buffer is not None:
            self.sg_buffer.add(state, goal, h_reward, next_state)
    
    def train_sg_policy(self, batch_size=64):
        if self.sg_buffer is None or len(self.sg_buffer) < batch_size:
            return
        
        s_t, g_t, r_t, ns_t = self.sg_buffer.sample(batch_size)
        s_t = s_t.to(self.device)
        g_t = g_t.to(self.device)
        r_t = r_t.to(self.device)
        ns_t = ns_t.to(self.device)
        
        with torch.no_grad():
            next_subgoal = self.sg_policy(ns_t)
            best_subgoals = self.get_subgoal_candidates(
                state=ns_t,
                subgoal=next_subgoal,
                n_candidates=self.subgoals_cand)    
            
            all_q_next = []
            for i in range(self.subgoals_cand):
                # i-th subgoal candidates
                cand = best_subgoals[:, i, :]
                q_val = self.sg_target(ns_t, cand)
                all_q_next.append(q_val)
            all_q_next = torch.stack(all_q_next, dim=0).transpose(0,1).squeeze(-1)
            
            # max Q in each batch
            q_next_max, _ = all_q_next.max(dim=1, keepdim=True)
            
            # TD target
            target_h = r_t + self.sg_gamma * q_next_max
            
        q_now = self.sg_critic(s_t, g_t)
        sg_critic_loss = F.mse_loss(q_now, target_h)
        
        self.sg_critic_optimizer.zero_grad()
        sg_critic_loss.backward()
        self.sg_critic_optimizer.step()
        
        with torch.no_grad():
            base_subgoal_s = self.sg_policy(s_t)
            best_subgoal_s = self.get_subgoal_candidates(
                state=s_t,
                subgoal=base_subgoal_s,
                n_candidates=self.subgoals_cand)
        
        # evaluate Q for each candidate subgoal
        all_q_vals = []
        for i in range(self.subgoals_cand):
            g_i = best_subgoal_s[:, i, :]
            q_i = self.sg_critic(s_t, g_i)
            all_q_vals.append(q_i)
        all_q_vals = torch.stack(all_q_vals, dim=0).transpose(0,1).squeeze(-1)
        
        # find best subgoal index (Max Q)
        best_indices = all_q_vals.argmax(dim=1)
        # compute Q of best subgoal in each batch
        best_subgoal = []
        for b in range(s_t.size(0)):
            idx = best_indices[b]
            best_subgoal.append(best_subgoal_s[b, idx, :].unsqueeze(0))
        best_subgoal = torch.cat(best_subgoal, dim=0)

        new_g = self.sg_policy(s_t)
        q_val = self.sg_critic(s_t, new_g)
        policy_loss = -q_val.mean()
        
        align_loss = F.mse_loss(new_g, best_subgoal.detach())
        
        sg_policy_loss = policy_loss + 1.0 * align_loss
            
        self.sg_optimizer.zero_grad()
        sg_policy_loss.backward()
        self.sg_optimizer.step()
        
        for param, tparam in zip(self.sg_critic.parameters(), self.sg_target.parameters()):
            tparam.data.copy_(self.tau * param.data + (1 - self.tau) * tparam.data)
    
    #-----------------------------------------------------------------------------#
    # Generate Multi-Step Action Sequence
    #-----------------------------------------------------------------------------#
    def sample_multi_step_actions(self, state, subgoal=None):
        
        state_seq = []
        action_seq = []
        
        current_state = state.clone()
        state_seq.append(current_state.unsqueeze(1))
        
        for t in range(self.horizon):
            if subgoal is not None:
                a_t = self.actor(current_state, subgoal)
            elif subgoal is None:
                zero_sub = torch.zeros_like(current_state)
                a_t = self.actor(current_state, zero_sub)
            else:
                a_t = self.actor(current_state)
            a_t = self.actor.clip_action(a_t)
            action_seq.append(a_t.unsqueeze(1))
            
            next_s, _, _ = self.dynamics.simulate_ensemble(current_state, a_t)
            current_state = next_s.clone()
            state_seq.append(current_state.unsqueeze(1))
            
        # concat along time dim
        action_seq = torch.cat(action_seq, dim=1)
        state_seq = torch.cat(state_seq, dim=1)
        return state_seq, action_seq
    
    #-----------------------------------------------------------------------------#
    # Compute Subgoal Reward
    #-----------------------------------------------------------------------------#
    def compute_subgoal_reward(self, last_state, subgoal, threshold=2.0):
        dist = (last_state - subgoal).pow(2).sum(dim=1).sqrt()
        high_reward = (dist < threshold).float()
        return high_reward
    

    def train(self, replay_buffer, iterations, batch_size=100):
        metric = {'bc_loss': [], 'actor_loss': [], 'critic_loss': [], 'subgoal_penalty': []}
        
        for _ in tqdm(range(iterations)):
            #-----------------------------------------------------------------------------#
            # (1) Critic Update (Learn)
            #-----------------------------------------------------------------------------#
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            current_q_values = self.critic(state, action)
            
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                
                zero_sub = torch.zeros_like(next_state_rpt)
                next_action_rpt = self.ema_model(next_state_rpt, subgoal=zero_sub)
                q_next_rpt = self.critic_target(next_state_rpt, next_action_rpt)
                q_next = q_next_rpt.view(batch_size, 10, -1).max(dim=1)[0]
            else:
                zero_sub = torch.zeros_like(next_state)
                next_action = self.ema_model(next_state, subgoal=zero_sub)
                q_next = self.critic_target(next_state, next_action)

            target_q = (reward + not_done * self.discount * q_next).detach()
            critic_loss = F.mse_loss(current_q_values, target_q) 

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()
            
            
            #-----------------------------------------------------------------------------#
            # (2) Generate Action Candidates (Subgoal Policy)
            #-----------------------------------------------------------------------------#
            subgoal = self.get_subgoal(state)

            #-----------------------------------------------------------------------------#
            # (3) Action Candidates (Diffusion Policy)
            #-----------------------------------------------------------------------------#
            # multi-step candidate actions
            state_seq, action_seq = self.sample_multi_step_actions(state, subgoal=subgoal)            
            last_state = state_seq[:, -1, :]
            # last_state = last_state.unsqueeze(1)
            sg_reward = self.compute_subgoal_reward(last_state, subgoal)
            self.save_sg_transition(state, subgoal, sg_reward.unsqueeze(1), last_state)

            #-----------------------------------------------------------------------------#
            # (5) Actor Update
            #-----------------------------------------------------------------------------#
            bc_loss, new_action = self.actor.loss(state, action, subgoal=subgoal)
            bc_new_action = self.actor.clip_action(new_action)
            q_ensembles = self.critic(state, bc_new_action)
            
            if self.pess_method == 'min':
                chosen_q_val = q_ensembles.min(dim=1, keepdim=True)[0]
            elif self.pess_method == 'lcb':
                mu = q_ensembles.mean(dim=1, keepdim=True)
                std = q_ensembles.std(dim=1, keepdim=True)
                chosen_q_val = mu - self.lcb_coef * std
            else:
                chosen_q_val = q_ensembles.mean(dim=1, keepdim=True)
            q_loss = -chosen_q_val.mean() / (q_ensembles.abs().mean().detach() + 1e-9)
            
            # subgoal penalty
            subgoal_penalty = (last_state - subgoal).pow(2).sum(dim=1).sqrt().mean()

            # Final Actor Loss
            actor_loss = bc_loss + self.eta * q_loss + self.subgoal_reg * subgoal_penalty
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            
            #-----------------------------------------------------------------------------#
            # EMA & Target Network Update
            #-----------------------------------------------------------------------------#
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1
            
            self.train_sg_policy(batch_size=batch_size)
            
            metric['bc_loss'].append(bc_loss.item())
            metric['actor_loss'].append(actor_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            metric['subgoal_penalty'].append(subgoal_penalty.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            subgoal = self.sg_policy(state)
            state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
            subgoal_rpt = torch.repeat_interleave(subgoal, repeats=50, dim=0)
            cand_actions = self.actor(state_rpt, subgoal=subgoal_rpt)
            q_ens = self.critic_target(state_rpt, cand_actions)
            
            if self.pess_method == 'lcb':
                mu = q_ens.mean(dim=1, keepdim=True)
                std = q_ens.std(dim=1, keepdim=True)
                final_q = mu - self.lcb_coef * std
            elif self.pess_method == 'min':
                final_q = q_ens.min(dim=1, keepdim=True)[0]
            else:
                final_q = q_ens.mean(dim=1, keepdim=True)
            
            probs = F.softmax(final_q.flatten(), dim=0)
            idx = torch.multinomial(probs, 1)
            chosen_action = cand_actions[idx].squeeze(0)
        return chosen_action.cpu().numpy()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))