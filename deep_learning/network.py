import torch.nn as nn
import torch.nn.functional as F
import torch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.audio_fc = nn.Sequential(nn.LayerNorm(192),
                                      nn.Linear(192, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 32))
        self.image_fc = nn.Sequential(nn.LayerNorm(512),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 32))

    def forward(self, pos_audios, pos_images, neg_images):
        norm_pos_audios, norm_pos_images, norm_neg_images = F.normalize(pos_audios, dim=0, p=2), F.normalize(pos_images,
                                                                                                             dim=0,
                                                                                                             p=2), F.normalize(
            neg_images, dim=0, p=2)
        return self.audio_fc(norm_pos_audios), self.image_fc(norm_pos_images), self.image_fc(norm_neg_images)