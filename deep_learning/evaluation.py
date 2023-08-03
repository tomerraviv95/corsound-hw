import torch


def calculate_cur_identification_accuracy(out_a, out_p, out_n):
    cur_correct_decisions = 0
    for a, p, n in zip(out_a, out_p, out_n):
        p_dist = torch.dist(a, p).item()
        n_dist = torch.dist(a, n).item()
        if p_dist < n_dist:
            cur_correct_decisions += 1
    cur_correct_decisions /= out_a.shape[0]
    return cur_correct_decisions


def evaluate(val_dataloader, net):
    net.eval()
    correct_decisions = 0
    for batch_idx, batch in enumerate(val_dataloader):
        pos_audios, pos_images, neg_images = batch  # anchor, positive, negative
        if torch.cuda.is_available():
            pos_audios, pos_images, neg_images = pos_audios.cuda(), pos_images.cuda(), neg_images.cuda()
        out_a, out_p, out_n = net(pos_audios, pos_images, neg_images)
        cur_correct_decisions = calculate_cur_identification_accuracy(out_a, out_p, out_n)
        correct_decisions += cur_correct_decisions
    correct_decisions /= (1 + batch_idx)
    print(f"Total correct decisions: {correct_decisions}")
