import gym
gym.logger.set_level(40)

import minerl
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import StateActionMap
from data import process_batch, BufferedBatchIter


if __name__ == "__main__":
    n_steps_between_frames = 8
    model = StateActionMap(state_features=128, n_steps=n_steps_between_frames).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_data = minerl.data.make('MineRLObtainDiamond-v0')
    eval_data = minerl.data.make('MineRLTreechop-v0')
    disc_loss = nn.BCELoss()
    cont_loss = nn.MSELoss()

    epochs = 10
    batch_size = 128

    for epoch in range(epochs):
        # Train
        model.train()
        iterator = DataLoader(BufferedBatchIter(train_data, n_steps_between_frames=n_steps_between_frames),
                              batch_size=None)
        sum_loss = 0.
        n_batches = 0
        for batch in iterator:
            image_arr, disc_targets, cont_targets = batch
            image_arr = torch.Tensor(image_arr).cuda()
            disc_targets = torch.Tensor(disc_targets).cuda()
            cont_targets = torch.Tensor(cont_targets).cuda()
            actions = torch.cat([disc_targets, cont_targets], dim=-1)

            disc_pred, cont_pred = model(image_arr, actions)
            loss = disc_loss(disc_pred, disc_targets) + cont_loss(cont_pred, cont_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            n_batches += 1
            print(f'Epoch {epoch}, batch {n_batches}: Train loss {loss.item()}', sum_loss / n_batches, end='\r')
        print('')

        # Eval
        model.eval()
        iterator = DataLoader(BufferedBatchIter(eval_data, n_steps_between_frames=n_steps_between_frames),
                              batch_size=None)
        sum_loss = 0.
        n_batches = 0
        for batch in iterator:
            image_arr, disc_targets, cont_targets = batch
            image_arr = torch.Tensor(image_arr).cuda()
            disc_targets = torch.Tensor(disc_targets).cuda()
            cont_targets = torch.Tensor(cont_targets).cuda()

            with torch.no_grad():
                disc_pred, cont_pred = model(image_arr, actions)
            loss = disc_loss(disc_pred, disc_targets) + cont_loss(cont_pred, cont_targets)

            sum_loss += loss.item()
            n_batches += 1
            print(f'Epoch {epoch}, batch {n_batches}: Eval loss', sum_loss / n_batches, end='\r')
        print('')

        torch.save(model.state_dict(), f'./model_smaller{epoch + 1}.pt')
