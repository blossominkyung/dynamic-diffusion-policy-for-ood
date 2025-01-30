import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EnsembleDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, num_ensembles=5):
        super().__init__()
        self.num_ensembles = num_ensembles
        
        self.models = nn.ModuleList([
            MLPModel(state_dim, action_dim, hidden_size) 
            for _ in range(num_ensembles)
        ])

    def forward(self, state, action):
        out_next_states = []
        out_rewards = []
        for m in self.models:
            n_s, r = m(state, action)
            out_next_states.append(n_s)
            out_rewards.append(r)

        next_states = torch.stack(out_next_states, dim=0)
        rewards = torch.stack(out_rewards, dim=0)
        return next_states, rewards

    def get_uncertainty(self, next_states):
        mean_ns = torch.mean(next_states, dim=0)
        var_ns = torch.mean((next_states - mean_ns.unsqueeze(0))**2, dim=(0,2))
        return var_ns


class MLPModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim + 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        next_state = out[..., :-1]
        reward = out[..., -1:]
        return next_state, reward


class Dynamics:
    def __init__(self,
                 state_dim,
                 action_dim,
                 ensemble_size=5,
                 hidden_size=256,
                 device='cpu',
                 alpha=0.5):
        
        self.device = device
        self.ensemble = EnsembleDynamics(state_dim, action_dim, 
                                         hidden_size, ensemble_size).to(device)
        
        self.ensemble_opt = optim.Adam(self.ensemble.parameters(), lr=1e-3)
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim

    def train_dynamics(self, replay_buffer, epochs=200, batch_size=100):
        for e in range(epochs):
            s, a, s_next, r, done = replay_buffer.sample(batch_size)
            s   = s.float().to(self.device)
            a   = a.float().to(self.device)
            s_n = s_next.float().to(self.device)
            r   = r.float().to(self.device)
            
            pred_next_states, pred_rewards = self.ensemble(s, a)
            
            loss_ns = F.mse_loss(pred_next_states, s_n.unsqueeze(0))
            loss_r  = F.mse_loss(pred_rewards, r.unsqueeze(0).squeeze(0))

            loss = loss_ns + loss_r

            self.ensemble_opt.zero_grad()
            loss.backward()
            self.ensemble_opt.step()
        
            if e % 2 == 0:
                print(f"[Dynamics Train] epoch: {e}, loss_ns: {loss_ns.item():.4f}, loss_r: {loss_r.item():.4f}, loss: {loss.item():.4f}")
                
    @torch.no_grad()
    def simulate_ensemble(self, state, action):
        pred_ns, pred_r = self.ensemble(state, action)

        # ensemble mean
        mean_ns = torch.mean(pred_ns, dim=0)
        mean_r  = torch.mean(pred_r, dim=0)

        # ensemble variance => penalty
        var_ns  = self.ensemble.get_uncertainty(pred_ns)
        penalty = self.alpha * var_ns.unsqueeze(-1)     
        return mean_ns, mean_r, penalty


    @torch.no_grad()
    def multistep_simulate_ensemble(self, state, action, aggregate="mean"):
        device = state.device
        batch_size = state.shape[0]
        H = action.shape[1]
        
        s = state
        all_rewards=[]
        all_penalties=[]
        
        for t in range(H):
            a_t = action[:, t, :]
            # 1) predict ensemble dynamics
            pred_nexs, pred_r = self.ensemble(s, a_t)
            # 2) ensemble mean/var
            mean_ns = torch.mean(pred_nexs, dim=0)
            mean_r  = torch.mean(pred_r, dim=0)
            var_ns  = self.ensemble.get_uncertainty(pred_nexs)
            # 3) compute penalty
            penalty_t = self.alpha * var_ns
            # 4) update s 
            s = mean_ns

            all_rewards.append(mean_r.squeeze(-1))
            all_penalties.append(penalty_t)
        
        all_rewards = torch.stack(all_rewards, dim=1)
        all_penalties = torch.stack(all_penalties, dim=1)
            
        if aggregate == "mean":
            total_reward = all_rewards.mean(dim=1)
            total_penalty = all_penalties.mean(dim=1)
        elif aggregate == "sum":
            total_reward = all_rewards.sum(dim=1)
            total_penalty = all_penalties.sum(dim=1)
        else:
            total_reward = all_rewards.min(dim=1)
            total_penalty = all_penalties.min(dim=1)
        
        return s, total_reward, total_penalty