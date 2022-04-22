from architectures import CLIPModel
from data import TransitionDataset

import minerl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


if __name__ == '__main__':
    writer = SummaryWriter()

    data_container = minerl.data.make('MineRLTreechop-v0')
    trajectory_names = data_container.get_trajectory_names()
    train_trajectory_names = trajectory_names[:80]
    eval_trajectory_names = trajectory_names[80: 100]
    train_dataset = TransitionDataset(data_container=data_container, trajectory_names=train_trajectory_names)
    eval_dataset = TransitionDataset(data_container=data_container, trajectory_names=eval_trajectory_names)

    epochs = 30
    batch_size = 256
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    target = torch.arange(0, batch_size).cuda()

    for model_name in ['mobilenet_pretrained', 'mobilenet_random', 'convnet']:
        model = CLIPModel(backbone_name=model_name).cuda()
        optim = torch.optim.Adam(model.parameters(), lr=0.003)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=0.0001)

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss_sum = 0.
            for batch in tqdm(train_dataloader):
                batch = batch.cuda()
                logits = model(batch)
                loss = criterion(logits, target) + criterion(logits.t(), target)
                train_loss_sum += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()

            # Eval
            model.eval()
            eval_loss_sum = 0.
            for batch in tqdm(eval_dataloader):
                batch = batch.cuda()
                with torch.no_grad():
                    logits = model(batch)
                loss = criterion(logits, target) + criterion(logits.t(), target)
                eval_loss_sum += loss.item()

            lr_scheduler.step()
            writer.add_scalars(model_name,
                               {'train': train_loss_sum / len(train_dataloader),
                                'eval': eval_loss_sum / len(eval_dataloader)},
                               epoch)
        writer.flush()
    writer.close()