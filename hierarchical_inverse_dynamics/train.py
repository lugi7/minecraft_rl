import minerl
import torch
from minerl.data import BufferedBatchIter

from torch import nn

from model import StateActionMap
from data import process_batch


if __name__ == "__main__":
    model = StateActionMap(state_features=256).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_data = minerl.data.make('MineRLObtainDiamond-v0')
    eval_data = minerl.data.make('MineRLTreechop-v0')
    disc_loss = nn.BCELoss()
    cont_loss = nn.MSELoss()

    epochs = 10
    batch_size = 32

    for epoch in range(epochs):
        # Train
        model.train()
        iterator = BufferedBatchIter(train_data)
        sum_loss = 0.
        n_batches = 0
        for batch in iterator.buffered_batch_iter(batch_size=batch_size, num_epochs=1):
            image_arr, disc_targets, cont_targets = process_batch(batch)
            image_arr = torch.Tensor(image_arr).cuda()
            disc_targets = torch.Tensor(disc_targets).cuda()
            cont_targets = torch.Tensor(cont_targets).cuda()

            disc_pred, cont_pred = model(image_arr)
            loss = disc_loss(disc_pred, disc_targets) + cont_loss(cont_pred, cont_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            n_batches += 1
            print(f'Epoch {epoch}: Train loss', sum_loss / n_batches)

        # Eval
        model.eval()
        iterator = BufferedBatchIter(eval_data)
        sum_loss = 0.
        n_batches = 0
        for batch in iterator.buffered_batch_iter(batch_size=batch_size, num_epochs=1):
            image_arr, disc_targets, cont_targets = process_batch(batch)
            image_arr = torch.Tensor(image_arr).cuda()
            disc_targets = torch.Tensor(disc_targets).cuda()
            cont_targets = torch.Tensor(cont_targets).cuda()

            with torch.no_grad():
                disc_pred, cont_pred = model(image_arr)
            loss = disc_loss(disc_pred, disc_targets) + cont_loss(cont_pred, cont_targets)

            sum_loss += loss.item()
            n_batches += 1
            print(f'Epoch {epoch}: Eval loss', sum_loss / n_batches)


