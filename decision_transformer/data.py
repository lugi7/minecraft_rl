from torch.utils.data import Dataset, DataLoader
import minerl
import numpy as np


class TrajectoryDataset(Dataset):
    def __init__(self, dataset_name, seq_len=256):
        super(TrajectoryDataset, self).__init__()
        self.data_container = minerl.data.make(dataset_name)
        self.trajectory_names = self.data_container.get_trajectory_names()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.trajectory_names)

    def __getitem__(self, item):
        trajectory_name = self.trajectory_names[item]
        traj = self.data_container.load_data(trajectory_name)
        sample = self._process_trajectory(traj)
        n_chunks = len(sample[0]) // self.seq_len
        choice = np.random.randint(0, n_chunks)
        start = choice * self.seq_len
        end = (choice + 1) * self.seq_len
        sample = tuple([x[start: end] for x in sample])
        return sample, (start, end)

    def _process_single_action(self, action):
        disc_action_names = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint']
        disc_action = np.stack([action[name] for name in disc_action_names], axis=0)
        cont_action = action['camera'].astype(np.float32) / 180.
        return np.concatenate([disc_action, cont_action])

    def _process_trajectory(self, traj):
        obss, actions, rewards, _, _ = zip(*traj)
        obss = np.stack([obs['pov'] for obs in obss]).astype(np.float32)
        actions = np.stack([self._process_single_action(action) for action in actions]).astype(np.float32)
        rtgs = np.cumsum(rewards[::-1])[::-1][:, None]
        rtgs = np.copy(rtgs).astype(np.float32)
        return obss, actions, rtgs


if __name__ == "__main__":
    data = TrajectoryDataset('MineRLTreechop-v0')
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    for batch in dataloader:
        obss, actions, rtgs = batch
        print(actions)
