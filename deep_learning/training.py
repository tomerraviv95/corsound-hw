import torch
from torch.utils.data import DataLoader

BATCH_PRINT = 25
ONE_HUNDRED = 100


def train(train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: torch.nn.modules.loss._Loss,
          net: torch.nn.Module, max_epoch: int):
    net.train()
    # run over all epochs
    for epoch in range(max_epoch):
        # for each epoch, run over all batches
        for batch_idx, batch in enumerate(train_dataloader):
            pos_audios, pos_images, neg_images = batch  # anchor, positive, negative
            if torch.cuda.is_available():
                pos_audios, pos_images, neg_images = pos_audios.cuda(), pos_images.cuda(), neg_images.cuda()
            optimizer.zero_grad()
            out_a, out_p, out_n = net(pos_audios, pos_images, neg_images)
            loss = loss_function(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            # print the loss every BATCH_PRINT batches
            if batch_idx % BATCH_PRINT == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(pos_audios), len(train_dataloader.dataset),
                           ONE_HUNDRED * batch_idx / len(train_dataloader), loss.item()))
