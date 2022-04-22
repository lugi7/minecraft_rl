import gym
import minerl
from tqdm import tqdm
import numpy as np


class Evaluator:
    def __init__(self, env_name, n_processes=1):
        self.env_name = env_name
        self.n_processes = n_processes
        self.multiprocessing = n_processes > 1
        if self.multiprocessing:
            raise NotImplementedError
        else:
            self.env = gym.make(self.env_name)
            # initial reset takes longer, so we might want to reset it asynchronously
            self.env.reset()

    def _eval_episode(self, policy):
        done = False
        step_counter = 0
        reward_arr = []
        obs = self.env.reset()
        while not done:
            action = policy(obs)
            obs, reward, done, _ = self.env.step(action)
            step_counter += 1
            reward_arr.append(reward)
        return reward_arr, step_counter

    def eval_policy(self, policy, n_runs=100):
        rewards = []
        episode_lens = []
        for _ in tqdm(range(n_runs)):
            reward_arr, step_counter = self._eval_episode(policy)
            rewards.append(sum(reward_arr))
            episode_lens.append(step_counter)
        return {'mean reward': np.mean(rewards),
                'mean episode length': np.mean(episode_lens)}


if __name__ == '__main__':
    evaluator = Evaluator(env_name='MineRLTreechop-v0')

    class DummyPolicy:
        def __init__(self):
            self.env = gym.make('MineRLTreechop-v0')

        def __call__(self, obs):
            return self.env.action_space.sample()

    dummy_policy = DummyPolicy()
    print(evaluator.eval_policy(dummy_policy))
