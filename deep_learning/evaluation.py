import torch
from torch import nn
from torch.utils.data import DataLoader


def calculate_identification_accuracy(out_a: torch.Tensor, out_p: torch.Tensor, out_n: torch.Tensor) -> int:
    """
    Calculates the measure of identification accuracy
    :param out_a: batch of anchor samples
    :param out_p: batch of positives samples
    :param out_n: batch of negatives samples
    :return: number of correct decisions, where dist(a,p) < dist(a,n) as desired
    """
    cur_correct_decisions = 0
    for a, p, n in zip(out_a, out_p, out_n):
        p_dist = torch.dist(a, p).item()
        n_dist = torch.dist(a, n).item()
        if p_dist < n_dist:
            cur_correct_decisions += 1
    cur_correct_decisions /= out_a.shape[0]
    return cur_correct_decisions


def evaluate(val_dataloader: DataLoader, net: nn.Module) -> float:
    net.eval()
    identification_acc = 0
    # calculate the measure over batches, aggregate at the end
    print('*' * 20)
    for batch_idx, batch in enumerate(val_dataloader):
        pos_audios, pos_images, neg_images = batch  # anchor, positive, negative
        if torch.cuda.is_available():
            pos_audios, pos_images, neg_images = pos_audios.cuda(), pos_images.cuda(), neg_images.cuda()
        out_a, out_p, out_n = net(pos_audios, pos_images, neg_images)
        cur_identification_acc = calculate_identification_accuracy(out_a, out_p, out_n)
        identification_acc += cur_identification_acc
    identification_acc /= (1 + batch_idx)
    print(f"Identification accuracy on the dataset: {identification_acc}")
    return identification_acc
