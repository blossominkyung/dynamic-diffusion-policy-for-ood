import time
import math
import torch
import numpy as np


class Data_Sampler(object):
    def __init__(self, data, device, reward_tune='no', max_size=1000000, cache_threshold=50000):

        self.device = device
        # 1) call offline data
        self.state = torch.from_numpy(data['observations']).float().to(self.device)
        self.action = torch.from_numpy(data['actions']).float().to(self.device)
        self.next_state = torch.from_numpy(data['next_observations']).float().to(self.device)
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float().to(self.device)
        self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float().to(self.device)

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        
        #-----------------------------------------------------------------------------#
        # 2) all offline data is is_rollout = False
        self.is_rollout = torch.zeros((self.size, 1), dtype=torch.float32).to(self.device)
        # dated buffer size
        self.max_size = max_size

        # (Optional) rollout 임시 캐시
        # 일정량 모아두었다가 한 번에 cat -> cat() 호출 횟수 감소
        self.cache_state = []
        self.cache_action = []
        self.cache_next_state = []
        self.cache_reward = []
        self.cache_not_done = []
        self.cache_is_rollout = []
        self.cache_threshold = cache_threshold
        #-----------------------------------------------------------------------------#

        # 3) reward tuning
        if reward_tune == 'normalize':
       	    reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        
        self.reward = reward
        self.reward_tune = reward_tune

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        return (
            self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind]
        )
        

def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
    reward /= (rt_max - rt_min)
    reward *= 1000.
    return reward