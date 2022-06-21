import gym
import minerl
import numpy as np
from collections import OrderedDict
import cv2

import torch

from model import DecisionTransformer
from mingpt.model import GPTConfig


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.max_seq_len = 256

    def _logits_to_action_dict(self, logits):
        actions_disc = torch.round(torch.sigmoid(logits[..., :-2])).cpu().numpy()
        actions_cont = torch.tanh(logits[..., -2:]).cpu().numpy() * 180.
        disc_action_names = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint']
        action_dict = OrderedDict()
        for ind, name in enumerate(disc_action_names):
            action_dict[name] = np.array(actions_disc[ind], dtype=np.int32)
        action_dict['camera'] = actions_cont
        self.last_action = action_dict
        return action_dict

    def _process_single_action(self, action):
        disc_action_names = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint']
        disc_action = np.stack([action[name] for name in disc_action_names], axis=0)
        cont_action = action['camera'].astype(np.float32) / 180.
        return np.concatenate([disc_action, cont_action])

    def init_step(self, obs, initial_rtg=64):
        self.rtgs = torch.ones((1, 1, 1)).cuda() * initial_rtg
        self.obss = torch.zeros((1, 1, 64, 64, 3)).cuda()
        self.obss[0, 0] = torch.Tensor(np.copy(obs['pov'])).cuda()
        self.actions = torch.zeros((1, 1, 10)).cuda()
        self.inds = (0, 1)
        with torch.no_grad():
            actions_pred = self.model(self.rtgs, self.obss, self.actions, self.inds)[0, -1]
        return self._logits_to_action_dict(actions_pred)

    def step(self, obs, reward):
        rtg = self.rtgs[:, -1:, :] - reward
        self.rtgs = torch.cat([self.rtgs, rtg], dim=1)[:, -self.max_seq_len:]
        obs = torch.Tensor(np.copy(obs['pov'])).cuda().view(1, 1, 64, 64, 3)
        self.obss = torch.cat([self.obss, obs], dim=1)[:, -self.max_seq_len:]
        last_action = torch.tensor(self._process_single_action(self.last_action)).cuda()
        self.actions[0, -1, :] = last_action
        self.actions = torch.cat([self.actions, torch.zeros_like(self.actions[:, -1:, :])], dim=1)[:, -self.max_seq_len:]
        self.inds = (max(0, self.inds[1] - self.max_seq_len), self.inds[1])
        with torch.no_grad():
            actions_pred = self.model(self.rtgs, self.obss, self.actions, self.inds)[0, -1]
        return self._logits_to_action_dict(actions_pred)


if __name__ == "__main__":
    env = gym.envs.make('MineRLTreechop-v0')
    seq_len = 256
    n_tiles_per_axis = 8

    config = GPTConfig(vocab_size=1,
                       block_size=seq_len * 3,
                       n_tiles_per_axis=n_tiles_per_axis,
                       window_size=8,
                       n_embd=128,
                       n_layer=4,
                       n_head=8,
                       seq_len=256)
    model = DecisionTransformer(config=config).cuda()
    model.load_state_dict(torch.load('./model_250.pt'))
    model.eval()
    wrapped_model = ModelWrapper(model=model)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 20, (64, 64))

    # Initial trajectory
    obs = env.reset()
    action = wrapped_model.init_step(obs=obs)
    done = False

    while not done:
        print(action)
        video.write(obs['pov'][..., ::-1])
        obs, reward, done, _ = env.step(action)
        action = wrapped_model.step(obs, reward)
    video.release()

