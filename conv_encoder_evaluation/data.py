import numpy as np
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, data_container, trajectory_names):
        super(TransitionDataset, self).__init__()
        trajectories = [data_container.load_data(tn) for tn in trajectory_names]
        self.trajectories = [sample for trajectory in trajectories for sample in trajectory]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        sarsd = self.trajectories[item]
        out = np.stack([sarsd[0]['pov'], sarsd[-2]['pov']], axis=0).astype(np.float32) / 255.
        return out
