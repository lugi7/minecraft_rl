from copy import deepcopy
import random

import numpy as np

from torch.utils.data import IterableDataset, DataLoader
import minerl
from minerl.data.util import multimap


def process_batch(batch):
    """
    Function for processing a raw batch from buffered_batch_iter to numeric arrays
    :param batch: input batch
    :return: image_arr: (bs * 2, 3, 64, 64), disc_targets: (bs, 8), cont_targets: (bs, 2)
    """
    states, actions, _, states_prime, _ = batch

    states = states['pov']
    states_prime = states_prime['pov']
    image_arr = np.concatenate([states, states_prime]) / 255.
    image_arr = image_arr.transpose(0, 3, 1, 2)

    disc_action_names = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint']

    bs = len(actions)
    n_steps = len(actions[0])
    disc_targets = np.empty((bs, n_steps, len(disc_action_names)))
    cont_targets = np.empty((bs, n_steps, 2))

    for i in range(bs):
        for j in range(n_steps):
            action = actions[i][j]
            disc_targets[i, j] = np.stack([action[name] for name in disc_action_names], axis=0)
            cont_targets[i, j] = action['camera'].astype(np.float32) / 180.

    return image_arr, disc_targets, cont_targets


def stack(*args):
    return np.stack(args)


class BufferedBatchIter(IterableDataset):
    def __init__(self,
                 data_pipeline,
                 buffer_target_size=50000,
                 batch_size=256,
                 n_steps_between_frames=1):
        super(BufferedBatchIter, self).__init__()
        self.data_pipeline = data_pipeline
        self.data_buffer = []
        self.buffer_target_size = buffer_target_size
        self.all_trajectories = self.data_pipeline.get_trajectory_names()
        self.available_trajectories = deepcopy(self.all_trajectories)
        self.batch_size = batch_size
        self.n_steps_between_frames = n_steps_between_frames  # 1 corresponds to the standard behavior

    def optionally_fill_buffer(self):
        """
        This method is run after every batch, but only actually executes a buffer
        refill and re-shuffle if more data is needed
        """
        buffer_updated = False

        # Add trajectories to the buffer if the remaining space is
        # greater than our anticipated trajectory size (in the form of the empirical average)
        while len(self.data_buffer) < self.buffer_target_size:
            if len(self.available_trajectories) == 0:
                return
            traj_to_load = self.available_trajectories.pop()
            data_list = list(self.data_pipeline.load_data(traj_to_load))
            obs_list, action_list, reward_list, next_obs_list, dones_list = list(zip(*data_list))

            for i in range(len(data_list) - self.n_steps_between_frames + 1):
                sample = (obs_list[i],
                          action_list[i: i + self.n_steps_between_frames],
                          reward_list[i],
                          next_obs_list[i + self.n_steps_between_frames - 1],
                          dones_list[i + self.n_steps_between_frames - 1])
                self.data_buffer.append(sample)

            buffer_updated = True
        if buffer_updated:
            random.shuffle(self.data_buffer)

    def get_batch(self, batch_size):
        """A simple utility method for constructing a return batch in the expected format"""
        ret_dict_list = []
        for _ in range(batch_size):
            data_tuple = self.data_buffer.pop()
            ret_dict = dict(obs=data_tuple[0],
                            act=data_tuple[1],
                            reward=data_tuple[2],
                            next_obs=data_tuple[3],
                            done=data_tuple[4])
            ret_dict_list.append(ret_dict)
        return multimap(stack, *ret_dict_list)

    def __iter__(self):
        while True:
            # Refill the buffer if we need to
            # (doing this before getting batch so it'll run on the first iteration)
            self.optionally_fill_buffer()
            ret_batch = self.get_batch(batch_size=self.batch_size)
            if len(self.data_buffer) < self.batch_size:
                return

            keys = ('obs', 'act', 'reward', 'next_obs', 'done')
            yield process_batch(tuple([ret_batch[key] for key in keys]))


if __name__ == '__main__':
    data = minerl.data.make('MineRLTreechop-v0')
    gen = BufferedBatchIter(data, n_steps_between_frames=4)
    sample = next(iter(gen))
    print([x.shape for x in sample])