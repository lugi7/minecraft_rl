import torch
from torch import nn
import minerl
from minerl.data import BufferedBatchIter

from mingpt.model import GPTConfig

from model import ViT

if __name__ == "__main__":
    # Init
    n_tiles_per_axis = 8
    window_size = 8
    batch_size = 256
    epochs = 10
    config = GPTConfig(vocab_size=1,
                       block_size=n_tiles_per_axis ** 2,
                       n_tiles_per_axis=n_tiles_per_axis,
                       window_size=8,
                       n_embd=128,
                       n_layer=6,
                       n_head=8)
    model = ViT(config=config).cuda()
    optim = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train_data = minerl.data.make('MineRLObtainDiamond-v0')
    train_iterator = BufferedBatchIter(train_data)
    eval_data = minerl.data.make('MineRLTreechop-v0')
    eval_iterator = BufferedBatchIter(eval_data)

    # Construct targets
    t = torch.arange(0, n_tiles_per_axis)
    t1 = t.view(-1, 1).expand(n_tiles_per_axis, n_tiles_per_axis)
    t2 = t.view(1, -1).expand(n_tiles_per_axis, n_tiles_per_axis)
    target_arr = torch.stack([t1, t2], dim=2).view(-1, n_tiles_per_axis ** 2, 2).tile(batch_size, 1, 1)

    for epoch in range(epochs):
        # Train loop
        model.train()
        sum_loss = 0.
        n_batches = 0
        for current_state, action, reward, next_state, done \
                in train_iterator.buffered_batch_iter(batch_size=batch_size, num_epochs=1):
            image_arr = torch.Tensor(current_state['pov']).cuda()
            pred = model(image_arr)
            loss = criterion(pred, target_arr)

            optim.zero_grad()
            loss.backward()
            optim.step()

            sum_loss += loss.item()
            n_batches += 1
            print(f'Epoch {epoch}, batch {n_batches}: Train loss {loss.item()}', sum_loss / n_batches, end='\r')
        print('')

        # Eval loop
        model.eval()
        sum_loss = 0.
        n_batches = 0
        for current_state, action, reward, next_state, done \
                in eval_iterator.buffered_batch_iter(batch_size=batch_size, num_epochs=1):
            image_arr = torch.Tensor(current_state['pov']).cuda()
            with torch.no_grad():
                pred = model(image_arr)
            loss = criterion(pred, target_arr)
            sum_loss += loss.item()
            n_batches += 1
            print(f'Epoch {epoch}, batch {n_batches}: Eval loss', sum_loss / n_batches, end='\r')
        print('')

        torch.save(model.state_dict(), f'./model_{epoch + 1}.pt')