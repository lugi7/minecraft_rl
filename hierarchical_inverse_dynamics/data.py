import numpy as np


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
    disc_targets = np.stack([actions[name] for name in disc_action_names], axis=1)

    cont_targets = actions['camera'].astype(np.float32) / 180.

    return image_arr, disc_targets, cont_targets
