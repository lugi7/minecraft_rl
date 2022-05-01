from data import TrajectoryDataset
from torch.utils.data import DataLoader
from torch import nn
from model import DecisionTransformer
from mingpt.model import GPTConfig
import torch


if __name__ == "__main__":
    epochs = 10
    seq_len = 256
    n_tiles_per_axis = 8
    data = TrajectoryDataset('MineRLTreechop-v0', seq_len=seq_len)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    config = GPTConfig(vocab_size=1,
                       block_size=seq_len * 3,
                       n_tiles_per_axis=n_tiles_per_axis,
                       window_size=8,
                       n_embd=128,
                       n_layer=4,
                       n_head=8,
                       seq_len=256)
    model = DecisionTransformer(config=config).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    print(sum([p.numel() for p in model.parameters()]))

    disc_criterion = nn.BCELoss()
    cont_criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Train loop
        model.train()
        sum_loss = 0.
        n_batches = 0

        for batch, inds in dataloader:
            inds = [x.item() for x in inds]
            obss, actions, rtgs = batch
            obss = obss.cuda()
            actions = actions.cuda()
            rtgs = rtgs.cuda()

            actions_disc = actions[..., :-2]
            actions_cont = actions[..., -2:]

            actions_pred = model(rtgs, obss, actions, inds)
            actions_pred_disc = torch.sigmoid(actions_pred[..., :-2])
            actions_pred_cont = torch.tanh(actions_pred[..., -2:])

            loss = disc_criterion(actions_pred_disc, actions_disc) + cont_criterion(actions_pred_cont, actions_cont)

            optim.zero_grad()
            loss.backward()
            optim.step()

            sum_loss += loss.item()
            n_batches += 1
            print(f'Epoch {epoch}, batch {n_batches}: Train loss {loss.item()}', sum_loss / n_batches, end='\n')
        print('')
