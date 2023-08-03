import torch


def train(train_dataloader, optimizer, loss_function, net, max_epoch):
    net.train()
    for epoch in range(max_epoch):
        for batch_idx, batch in enumerate(train_dataloader):
            pos_audios, pos_images, neg_images = batch  # anchor, positive, negative
            if torch.cuda.is_available():
                pos_audios, pos_images, neg_images = pos_audios.cuda(), pos_images.cuda(), neg_images.cuda()
            optimizer.zero_grad()
            out_a, out_p, out_n = net(pos_audios, pos_images, neg_images)
            loss = loss_function(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            if batch_idx % 25 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(pos_audios), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader), loss.item()))
