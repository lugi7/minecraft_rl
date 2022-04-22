import numpy as np
import minerl
import torch
from torch.utils.data.dataset import IterableDataset


class DemonstrationDataset(IterableDataset):
    def __init__(self, env_name,
                 batch_size=32,
                 seq_len=256,
                 eval_size=32,
                 noise_seed=42,
                 noise_dim=16):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eval_size = eval_size
        self.noise_seed = noise_seed
        self.noise_dim = noise_dim

        self.data = minerl.data.make(env_name)
        self.trajectory_names = self.data.get_trajectory_names()
        self.trajectories = self._init_trajectories()
        self.trajectory_lens = {}
        self.noise_arr = {}

    def _init_trajectories(self):
        train_trajectories = [[x for x in self.data.load_data(k)]
                              for k in self.trajectory_names[:-self.eval_size]]
        eval_trajectories = [[x for x in self.data.load_data(k)]
                             for k in self.trajectory_names[-self.eval_size:]]
        trajectories_dict = {'train': train_trajectories,
                             'eval': eval_trajectories}
        return trajectories_dict

    def _init_noise_data(self):
        rng = np.random.default_rng(self.noise_seed)
        noise_train = [rng.normal(size=(trajectory_len, self.noise_dim))
                       for trajectory_len in self.trajectories['train']]
        noise_eval = [rng.normal(size=(trajectory_len, self.noise_dim))
                      for trajectory_len in self.trajectories['eval']]
        noise_dict = {'train': noise_train,
                      'eval': noise_eval}
        return noise_dict

    def _init_epoch_master_list(self):
        concat_batch_len = np.zeros((self.batch_size,), dtype=np.int32)
        trajectory_inds = list(range(len(trajectories)))
        np.random.shuffle(trajectory_inds)

        batch_master_list = [[] for i in range(batch_size)]
        trajectory_lens = [len(t) for t in trajectories]
        n_sequences_per_trajectory = {k: np.ceil(l / seq_len).astype(np.int32) for k, l in
                                      enumerate(trajectory_lens)}

        while trajectory_inds:
            min_inds = np.where(concat_batch_len == concat_batch_len.min())[0]
            for ind in min_inds:
                if not trajectory_inds:
                    break
                ti = trajectory_inds.pop()
                n_seqs = n_sequences_per_trajectory[ti]
                # construct tuples
                entries = []
                for n in range(n_seqs):
                    valid_steps = min(trajectory_lens[ti] - n * seq_len, seq_len)
                    entry = (ti, n, valid_steps)
                    entries.append(entry)
                batch_master_list[ind].extend(entries)
                concat_batch_len[ind] += n_seqs

        gen_len = max([len(x) for x in batch_master_list])

        return batch_master_list, gen_len

    def get_train_generator(self):
        pass

    def get_eval_generator(self):
        pass


if __name__ == '__main__':
    dataset = DemonstrationDataset('MineRLTreechop-v0')
